import atexit
from dataclasses import fields
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import pickle
import os
import torch
from nano_pearl.pearl_config import PEARLConfig
from nano_pearl.pearl_engine.pearl_model_runner import DraftModelRunner, TargetModelRunner
from nano_pearl.utils.pearl_logger import logger
from multiprocessing.synchronize import Event
from nano_pearl.pearl_engine.sequence import Sequence
from nano_pearl.layers.sampler import SamplingParams


class Controller:
    def __init__(self, config: PEARLConfig, control_event: Event):
        self.config = config
        self.draft_event = []
        self.target_event = []
        self.control_event = control_event
        self.draft_shm = SharedMemory(name=config.draft_config.group_name, create=True, size=2**20)
        self.target_shm = SharedMemory(name=config.target_config.group_name, create=True, size=2**20)

    def add_event(self, rank, event):
        if rank in self.config.draft_config.devices:
            self.draft_event.append(event)
        else:
            self.target_event.append(event)

    def write_draft_shm(self, method_name, *args):
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.draft_shm.buf[0:4] = n.to_bytes(4, "little")
        self.draft_shm.buf[4:n+4] = data
        for event in self.draft_event:
            event.set()
        
    def write_target_shm(self, method_name, *args):
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.target_shm.buf[0:4] = n.to_bytes(4, "little")
        self.target_shm.buf[4:n+4] = data
        for event in self.target_event:
            event.set()
    
    def read_output(self):
        n = int.from_bytes(self.target_shm.buf[0:4], "little")
        data = self.target_shm.buf[4:n+4]
        output, elapsed_time = pickle.loads(data)
        return output, elapsed_time


class PEARLEngine:    
    def __init__(self, config: PEARLConfig):
        self.config = config
        self.ps = []
        
        ctx = mp.get_context("spawn")
        # the control event is used to wait for the sub-processes to be ready
        self.control_event = ctx.Event()
        self.controller = Controller(config, self.control_event)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.draft_config.model, use_fast=True)
        config.eos = self.config.draft_config.eos
        logger.info(f"[Main Process] EOS token id: {config.eos}, EOS tokens: {self.tokenizer.decode(config.eos)}")   

        for i in range(config.world_size):
            event = ctx.Event()
            process = ctx.Process(target=DraftModelRunner if i in config.draft_config.devices else TargetModelRunner, args=(config, i, event, self.control_event))
            process.daemon = True        
            process.start()
            self.ps.append(process)
            self.controller.add_event(i, event)
        
        # wait for the initialization of the draft and target TP models
        logger.info("[Main Process] Waiting for the initialization of the draft and target TP models...", color="red")
        self.control_event.wait()
        self.control_event.clear()
        
        atexit.register(self.exit)
    

    def log(self, content: str):
        logger.info(f"[Main Process] Running log function, waiting for the sub-processes", color="red")
        self.controller.write_draft_shm("log", content)
        self.controller.write_target_shm("log", content)
        self.control_event.wait()
        self.control_event.clear()

    def run_model(self, seqs: list[Sequence], is_prefill: bool):        
        self.controller.write_draft_shm("run_model", seqs, is_prefill)
        self.controller.write_target_shm("run_model", seqs, is_prefill)
        self.control_event.wait()
        self.control_event.clear()

    def exit(self):
        self.controller.write_draft_shm("exit")
        self.controller.write_target_shm("exit")
        for p in self.ps:
            p.join()     
            if p.is_alive():
                p.kill()
                p.join()                
        self.controller.draft_shm.close()
        self.controller.target_shm.close()
        self.controller.draft_shm.unlink()
        self.controller.target_shm.unlink()
        

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.controller.write_draft_shm("add_request", seq)
        self.controller.write_target_shm("add_request", seq)
        self.control_event.wait()
        self.control_event.clear()
    
    def generate(self):
        self.controller.write_draft_shm("pearl_generate")
        self.controller.write_target_shm("pearl_generate")
        self.control_event.wait()
        self.control_event.clear()

        output, time = self.controller.read_output()
        output = sorted(output, key=lambda x: x[0])
        seq_id, token_ids, num_acc_tokens = zip(*output)
        output_text = [self.tokenizer.decode(token_ids, skip_special_tokens=False) for token_ids in token_ids]
        num_tokens = [len(t) for t in token_ids]
        
        return output_text, num_tokens, num_acc_tokens, time

    def AR_generate(self):
        """Only use target model for Auto-Regressive generation."""
        self.controller.write_draft_shm("parallel_generate")
        self.controller.write_target_shm("parallel_generate")
        self.control_event.wait()
        self.control_event.clear()

        output, time = self.controller.read_output()
        output = sorted(output, key=lambda x: x[0])
        seq_id, token_ids, _ = zip(*output)
        output_text = [self.tokenizer.decode(token_ids, skip_special_tokens=False) for token_ids in token_ids]
        num_tokens = [len(t) for t in token_ids]

        return output_text, num_tokens, None, time
    
    def bench_generate(self, num_pearl_steps: int = 100):
        self.controller.write_draft_shm("pearl_bench_generate", num_pearl_steps)
        self.controller.write_target_shm("pearl_bench_generate", num_pearl_steps)
        self.control_event.wait()
        self.control_event.clear()

        output, time = self.controller.read_output()
        output = sorted(output, key=lambda x: x[0])
        seq_id, token_ids, num_acc_tokens = zip(*output)
        output_text = [self.tokenizer.decode(token_ids, skip_special_tokens=False) for token_ids in token_ids]
        num_tokens = [len(t) for t in token_ids]

        return output_text, num_tokens, num_acc_tokens, time

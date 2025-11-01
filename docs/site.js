// Add 'scrolled' class to nav after slight scroll
(function(){
  var nav = document.querySelector('.nav');
  function onScroll(){
    if(!nav) return;
    if(window.scrollY > 8){ nav.classList.add('scrolled'); }
    else{ nav.classList.remove('scrolled'); }
  }
  window.addEventListener('scroll', onScroll, {passive:true});
  onScroll();

  // Simple reveal-on-scroll for elements with .reveal
  var els = Array.from(document.querySelectorAll('.reveal'));
  if('IntersectionObserver' in window){
    var io = new IntersectionObserver(entries => {
      entries.forEach(e => { if(e.isIntersecting){ e.target.classList.add('show'); io.unobserve(e.target); } });
    }, {rootMargin:'0px 0px -10% 0px'});
    els.forEach(el => io.observe(el));
  }else{
    els.forEach(el => el.classList.add('show'));
  }

  // Minimal Python syntax highlighter (safe placeholders -> escape -> spans)
  try{
    var pyBlocks = document.querySelectorAll('pre > code.language-python');
    if(pyBlocks.length){
      var kwRe = /(\b(?:def|class|return|import|from|as|if|elif|else|for|while|try|except|finally|with|yield|lambda|None|True|False|and|or|not|in|is|pass|break|continue|global|nonlocal|assert|raise)\b)/g;
      var defName = /\b(def|class)\s+(\w+)/g;
      var deco = /^(\s*)@(\w+)/gm;
      var strRe = /('''|""")[\s\S]*?\1|'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*"/g;
      var comRe = /(^|[^\\])#.*/gm;
      var numRe = /\b\d+(?:\.\d+)?\b/g;

      function esc(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

      pyBlocks.forEach(function(codeEl){
        var src = codeEl.textContent;

        // Stage 1: wrap tokens with placeholders
        // Strings and comments first (to avoid inner replacements)
        src = src.replace(strRe, function(m){ return '\u0091S'+m+'\u0092'; });
        src = src.replace(comRe, function(m, p1){ return p1+'\u0091C'+m.slice(p1.length)+'\u0092'; });
        // Decorators
        src = src.replace(deco, function(_, sp, name){ return sp+'\u0091D@'+name+'\u0092'; });
        // def/class + name
        src = src.replace(defName, function(_, kw, name){ return '\u0091K'+kw+'\u0092 \u0091F'+name+'\u0092'; });
        // Remaining keywords
        src = src.replace(kwRe, function(_, kw){ return '\u0091K'+kw+'\u0092'; });
        // Numbers
        src = src.replace(numRe, function(m){ return '\u0091N'+m+'\u0092'; });

        // Stage 2: escape whole string
        src = esc(src);

        // Stage 3: replace placeholders with spans
        src = src
          .replace(/\u0091K([\s\S]*?)\u0092/g, '<span class="tok-k">$1</span>')
          .replace(/\u0091F([\s\S]*?)\u0092/g, '<span class="tok-fn">$1</span>')
          .replace(/\u0091D([\s\S]*?)\u0092/g, '<span class="tok-dec">$1</span>')
          .replace(/\u0091S([\s\S]*?)\u0092/g, '<span class="tok-s">$1</span>')
          .replace(/\u0091C([\s\S]*?)\u0092/g, '<span class="tok-c">$1</span>')
          .replace(/\u0091N([\s\S]*?)\u0092/g, '<span class="tok-n">$1</span>');

        codeEl.innerHTML = src;
      });
    }
  }catch(e){ /* no-op */ }

  // Minimal Bash/Shell syntax highlighter (safe placeholders -> escape -> spans)
  try{
    var shBlocks = document.querySelectorAll('pre > code.language-bash, pre > code.language-shell, pre > code.language-sh, pre > code.language-zsh');
    if(shBlocks.length){
      var strRe = /('''|""")[\s\S]*?\1|'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*"/g;
      var comRe = /(^|[^\\])#.*/gm;
      var varRe = /\$(?:\w+|\{[^}]+\}|\([^)]+\))/g;
      var optRe = /(\s)(--[\w-]+|-[\w]+)/g;
      var numRe = /\b\d+(?:\.\d+)?\b/g;

      function esc(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

      shBlocks.forEach(function(codeEl){
        var src = codeEl.textContent;
        // Placeholders
        src = src.replace(strRe, function(m){ return '\u0091S'+m+'\u0092'; });
        src = src.replace(comRe, function(m, p1){ return p1+'\u0091C'+m.slice(p1.length)+'\u0092'; });
        src = src.replace(/^(\s*)([A-Za-z0-9_.-]+)(?=\s|$)/gm, function(_, sp, cmd){ return sp+'\u0091X'+cmd+'\u0092'; });
        src = src.replace(optRe, function(_, sp, opt){ return sp+'\u0091O'+opt+'\u0092'; });
        src = src.replace(varRe, function(m){ return '\u0091V'+m+'\u0092'; });
        src = src.replace(numRe, function(m){ return '\u0091N'+m+'\u0092'; });

        src = esc(src)
          .replace(/\u0091X([\s\S]*?)\u0092/g, '<span class="tok-cmd">$1</span>')
          .replace(/\u0091O([\s\S]*?)\u0092/g, '<span class="tok-opt">$1</span>')
          .replace(/\u0091V([\s\S]*?)\u0092/g, '<span class="tok-var">$1</span>')
          .replace(/\u0091S([\s\S]*?)\u0092/g, '<span class="tok-s">$1</span>')
          .replace(/\u0091C([\s\S]*?)\u0092/g, '<span class="tok-c">$1</span>')
          .replace(/\u0091N([\s\S]*?)\u0092/g, '<span class="tok-n">$1</span>');

        codeEl.innerHTML = src;
      });
    }
  }catch(e){ /* no-op */ }
})();

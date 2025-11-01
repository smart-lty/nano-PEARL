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
})();
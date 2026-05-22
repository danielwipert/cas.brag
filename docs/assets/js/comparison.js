// BRAG — side-by-side comparison reveal (section 06).
// Pre-baked. No frameworks. Works without JS too — CSS keeps stages visible
// when this script doesn't load (only the data-progressive attribute hides them).

(function () {
  'use strict';

  const root = document.querySelector('[data-compare]');
  if (!root) return;

  // Mark JS as present — CSS uses this to hide stages on initial paint
  document.documentElement.classList.add('js');

  const stages = Array.from(root.querySelectorAll('.compare-stage'));
  const btnNext = root.querySelector('[data-action="next"]');
  const btnAll = root.querySelector('[data-action="all"]');
  const btnReset = root.querySelector('[data-action="reset"]');
  const counter = root.querySelector('[data-counter]');

  // Group stages by reveal step (data-step). Step 1 is shown by default.
  const stepsMap = new Map();
  stages.forEach((el) => {
    const step = Number(el.dataset.step || 1);
    if (!stepsMap.has(step)) stepsMap.set(step, []);
    stepsMap.get(step).push(el);
  });
  const totalSteps = Math.max(...stepsMap.keys());
  let currentStep = 1;

  function hideFrom(step) {
    stages.forEach((el) => {
      const s = Number(el.dataset.step || 1);
      if (s > step) {
        el.hidden = true;
        el.classList.remove('is-revealed');
      } else if (s === step) {
        el.hidden = false;
        // first-paint stages skip animation
        if (step > 1) el.classList.add('is-revealed');
      } else {
        el.hidden = false;
      }
    });
  }

  function updateControls() {
    if (counter) counter.textContent = `Stage ${currentStep} of ${totalSteps}`;
    if (btnNext) {
      const done = currentStep >= totalSteps;
      btnNext.disabled = done;
      btnNext.textContent = done ? 'All stages revealed' : 'Reveal next stage';
      if (!done) {
        const arrow = document.createElement('span');
        arrow.className = 'arrow';
        arrow.textContent = '→';
        btnNext.appendChild(arrow);
      }
    }
    if (btnReset) btnReset.hidden = currentStep <= 1;
    if (btnAll) btnAll.hidden = currentStep >= totalSteps;
  }

  function showStep(step) {
    currentStep = Math.max(1, Math.min(totalSteps, step));
    hideFrom(currentStep);
    updateControls();
  }

  function showAll() {
    showStep(totalSteps);
  }

  function reset() {
    showStep(1);
    // scroll back into view for context
    root.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  if (btnNext) btnNext.addEventListener('click', () => showStep(currentStep + 1));
  if (btnAll) btnAll.addEventListener('click', showAll);
  if (btnReset) btnReset.addEventListener('click', reset);

  // Keyboard: arrow keys to step through when comparison is in view
  document.addEventListener('keydown', (e) => {
    const rect = root.getBoundingClientRect();
    const inView = rect.top < window.innerHeight && rect.bottom > 0;
    if (!inView) return;
    if (e.key === 'ArrowRight' || e.key === ' ') {
      if (currentStep < totalSteps) { e.preventDefault(); showStep(currentStep + 1); }
    } else if (e.key === 'ArrowLeft') {
      if (currentStep > 1) { e.preventDefault(); showStep(currentStep - 1); }
    }
  });

  showStep(1);
})();

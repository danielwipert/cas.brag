// share.js — copy deep links to per-principle mini-pages.
// Each .principle-share button has a data-url attribute pointing to its
// mini-page; clicking it copies the absolute URL to the clipboard so the
// principle can be shared on its own (with its own OG card).

(function () {
  'use strict';
  const buttons = document.querySelectorAll('.principle-share');
  if (!buttons.length) return;

  buttons.forEach((btn) => {
    btn.addEventListener('click', async () => {
      const path = btn.dataset.url;
      if (!path) return;
      const absolute = new URL(path, location.href).href;
      const original = btn.textContent;

      const confirm = () => {
        btn.textContent = 'Copied';
        btn.classList.add('copied');
        clearTimeout(btn._restoreTimer);
        btn._restoreTimer = setTimeout(() => {
          btn.textContent = original;
          btn.classList.remove('copied');
        }, 1600);
      };

      try {
        await navigator.clipboard.writeText(absolute);
        confirm();
      } catch (e) {
        // Fallback for older browsers / non-secure contexts:
        // create a temporary input, select, copy, remove.
        const input = document.createElement('input');
        input.value = absolute;
        document.body.appendChild(input);
        input.select();
        try {
          document.execCommand('copy');
          confirm();
        } catch (e2) {
          window.prompt('Copy this link:', absolute);
        }
        input.remove();
      }
    });
  });
})();

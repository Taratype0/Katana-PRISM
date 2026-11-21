(() => {
  const $ = id => document.getElementById(id);
  const figures = [...document.querySelectorAll('.exhibit')];
  const chapters = [...document.querySelectorAll('section.chapter')];
  const settings = JSON.parse($('report-settings')?.textContent || '{}');
  let eventSet = settings.event_set || 'stable_significant';
  let expandAll = false;
  const initialComparison = settings.initial_comparison || $('contrastSelect')?.options[0]?.value || 'all';
  if ($('contrastSelect')) $('contrastSelect').value = initialComparison;

  const scHierarchy = settings.sc_hierarchy || null;
  let activeExpContrast = scHierarchy ? (scHierarchy.initial_exp || Object.keys(scHierarchy.by_exp || {})[0] || '') : '';
  let activeSubtype = scHierarchy ? (scHierarchy.initial_subtype || '__ALL__') : '__ALL__';

  function resolveTwoTierContrast() {
    if (!scHierarchy || !scHierarchy.by_exp) return $('contrastSelect')?.value || initialComparison;
    const mapping = scHierarchy.by_exp[activeExpContrast] || {};
    if (mapping[activeSubtype]) return mapping[activeSubtype];
    if (mapping['__ALL__']) {
      activeSubtype = '__ALL__';
      return mapping['__ALL__'];
    }
    const firstKey = Object.keys(mapping)[0];
    if (firstKey) {
      activeSubtype = firstKey;
      return mapping[firstKey];
    }
    return $('contrastSelect')?.value || initialComparison;
  }

  function renderSubtypeControls() {
    if (!scHierarchy || !scHierarchy.by_exp) return;
    const mapping = scHierarchy.by_exp[activeExpContrast] || {};
    const keys = Object.keys(mapping);
    const sel = $('subtypeSelect');
    const pillsWrap = $('subtypePills');
    if (sel) {
      sel.innerHTML = '';
      for (const k of keys) {
        const opt = document.createElement('option');
        opt.value = k;
        opt.textContent = k === '__ALL__' ? 'All Cells (Experiment-wide)' : k.replace(/_/g, ' ');
        sel.appendChild(opt);
      }
      if (!keys.includes(activeSubtype)) activeSubtype = keys[0] || '__ALL__';
      sel.value = activeSubtype;
    }
    if (pillsWrap) {
      pillsWrap.innerHTML = '';
      for (const k of keys) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'subtype-pill' + (!expandAll && k === activeSubtype ? ' active' : '');
        btn.dataset.subtypeValue = k;
        btn.textContent = k === '__ALL__' ? 'All Cells (Experiment)' : k.replace(/_/g, ' ');
        btn.addEventListener('click', () => {
          expandAll = false;
          activeSubtype = k;
          if ($('subtypeSelect')) $('subtypeSelect').value = k;
          if ($('contrastSelect')) $('contrastSelect').value = resolveTwoTierContrast();
          applyScope();
        });
        pillsWrap.appendChild(btn);
      }
    }
    if ($('expContrastSelect')) $('expContrastSelect').value = activeExpContrast;
  }

  const policySelectors = [...document.querySelectorAll('.das-select')];
  policySelectors.forEach(select => {
    select.value = eventSet;
    select.addEventListener('change', () => {
      eventSet = select.value;
      policySelectors.forEach(s => { s.value = eventSet; });
      applyScope();
    });
  });

  if (location.protocol === 'file:') {
    document.querySelectorAll('a[href="REPORT.zip"]').forEach(a => { a.hidden = true; });
  }

  function syncPills(contrast) {
    document.querySelectorAll('.comparison-pill').forEach(btn => {
      if (btn.dataset.expValue) {
        btn.classList.toggle('active', !expandAll && btn.dataset.expValue === activeExpContrast);
      } else {
        btn.classList.toggle('active', !expandAll && btn.dataset.contrastValue === contrast);
      }
    });
    document.querySelectorAll('.subtype-pill').forEach(btn => {
      btn.classList.toggle('active', !expandAll && btn.dataset.subtypeValue === activeSubtype);
    });
    const expandBtn = $('expandAllScope');
    if (expandBtn) {
      expandBtn.classList.toggle('active-pill', expandAll);
      expandBtn.textContent = expandAll ? 'Focused comparison view' : 'Show all comparisons';
    }
  }

  function applyScope() {
    if (scHierarchy && $('contrastSelect')) {
      $('contrastSelect').value = resolveTwoTierContrast();
      renderSubtypeControls();
    }
    const contrast = expandAll ? 'all' : ($('contrastSelect')?.value || 'all');
    const celltype = $('celltypeSelect')?.value || 'all';
    let n = 0;
    for (const figure of figures) {
      const c = figure.dataset.contrast || 'all';
      const t = figure.dataset.celltype || 'all';
      const group = JSON.parse(figure.dataset.comparisons || '[]');
      const policy = figure.dataset.policy || 'common';
      const show = (contrast === 'all' || c === contrast || (c === 'all' && (!group.length || group.includes(contrast))))
        && (celltype === 'all' || t === 'all' || t === celltype)
        && (policy === 'common' || policy === eventSet);
      figure.classList.toggle('scope-hidden', !show);
      if (show) n++;
      for (const frame of figure.querySelectorAll('iframe[data-src]')) {
        if (show && !frame.getAttribute('src')) frame.src = frame.dataset.src;
        if (!show && frame.getAttribute('src')) frame.removeAttribute('src');
      }
    }
    chapters.forEach(ch => {
      ch.classList.toggle(
        'no-visible-figures',
        !!ch.querySelector('.exhibit') && ![...ch.querySelectorAll('.exhibit')].some(f => !f.classList.contains('scope-hidden'))
      );
    });
    if ($('scopeSummary')) {
      $('scopeSummary').textContent = expandAll
        ? `${n} exhibits · all declared comparisons displayed`
        : `${n} exhibits · active comparison & study-wide QC`;
    }
    syncPills($('contrastSelect')?.value || initialComparison);
    window.visibleReportFigures = n;
    window.reportEventSet = eventSet;
    window.reportReady = true;
    window.dispatchEvent(new CustomEvent('katana:scope-change', {
      detail: {
        contrast: $('contrastSelect')?.value || initialComparison,
        eventSet,
        expandAll,
      },
    }));
    updateNavigation();
  }

  addEventListener('message', ev => {
    const msg = ev.data || {};
    if (msg.type === 'katana-iframe-height' && typeof msg.height === 'number') {
      document.querySelectorAll('iframe.network-frame').forEach(fr => {
        if (fr.contentWindow === ev.source) {
          fr.style.height = `${Math.max(420, Math.min(2200, Math.ceil(msg.height) + 12))}px`;
        }
      });
    } else if (msg.type === 'katana-open-genecard' && msg.gene) {
      window.dispatchEvent(new CustomEvent('katana:open-genecard', {
        detail: { gene: String(msg.gene), eventId: String(msg.eventId || '') },
      }));
      requestAnimationFrame(() => {
        const target = $('kwb-dossier-card') || $('interactive-candidate-gene-workbench') || $('chapter-8');
        if (target) target.scrollIntoView({ behavior: 'auto', block: 'start' });
      });
    }
  });

  for (const id of ['contrastSelect', 'celltypeSelect']) {
    $(id)?.addEventListener('change', () => {
      expandAll = false;
      applyScope();
    });
  }

  $('expContrastSelect')?.addEventListener('change', e => {
    expandAll = false;
    activeExpContrast = e.target.value;
    if ($('contrastSelect')) $('contrastSelect').value = resolveTwoTierContrast();
    applyScope();
  });

  $('subtypeSelect')?.addEventListener('change', e => {
    expandAll = false;
    activeSubtype = e.target.value;
    if ($('contrastSelect')) $('contrastSelect').value = resolveTwoTierContrast();
    applyScope();
  });

  document.querySelectorAll('.comparison-pill').forEach(btn => {
    btn.addEventListener('click', () => {
      expandAll = false;
      if (btn.dataset.expValue) {
        activeExpContrast = btn.dataset.expValue;
        if ($('expContrastSelect')) $('expContrastSelect').value = activeExpContrast;
        if ($('contrastSelect')) $('contrastSelect').value = resolveTwoTierContrast();
      } else if ($('contrastSelect')) {
        $('contrastSelect').value = btn.dataset.contrastValue || initialComparison;
      }
      applyScope();
    });
  });

  $('expandAllScope')?.addEventListener('click', () => {
    expandAll = !expandAll;
    applyScope();
  });

  $('resetScope')?.addEventListener('click', () => {
    expandAll = false;
    if (scHierarchy) {
      activeExpContrast = scHierarchy.initial_exp || Object.keys(scHierarchy.by_exp || {})[0] || '';
      activeSubtype = scHierarchy.initial_subtype || '__ALL__';
      if ($('contrastSelect')) $('contrastSelect').value = resolveTwoTierContrast();
    } else if ($('contrastSelect')) {
      $('contrastSelect').value = initialComparison;
    }
    if ($('celltypeSelect')) $('celltypeSelect').value = 'all';
    applyScope();
  });

  $('printReport')?.addEventListener('click', () => print());

  document.querySelectorAll('.plot-button').forEach(button => {
    button.addEventListener('click', () => {
      const im = button.querySelector('img');
      if (!im) return;
      $('modalTitle').textContent = im.alt;
      $('modalImage').src = im.src;
      $('modalImage').alt = im.alt;
      $('downloadFigure').href = im.src;
      $('modalScroll').classList.remove('actual');
      $('figureModal').showModal();
    });
  });

  $('fitFigure')?.addEventListener('click', () => $('modalScroll').classList.remove('actual'));
  $('actualFigure')?.addEventListener('click', () => $('modalScroll').classList.add('actual'));
  $('closeFigure')?.addEventListener('click', () => $('figureModal').close());
  $('figureModal')?.addEventListener('click', e => {
    if (e.target === $('figureModal')) $('figureModal').close();
  });

  function updateNavigation() {
    let selected = chapters[0];
    for (const chapter of chapters) {
      if (chapter.getBoundingClientRect().top <= 165) selected = chapter;
      else break;
    }
    document.querySelectorAll('.toc a').forEach(a => {
      a.classList.toggle('active', a.getAttribute('href') === '#' + selected?.id);
    });
  }

  let scheduled = false;
  addEventListener('scroll', () => {
    if (!scheduled) {
      scheduled = true;
      requestAnimationFrame(() => { scheduled = false; updateNavigation(); });
    }
  }, { passive: true });
  addEventListener('resize', updateNavigation);
  addEventListener('load', updateNavigation);
  applyScope();
})();

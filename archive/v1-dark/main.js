/* ============================================================
   SeMoCo — project page behaviour
   ============================================================ */
(function () {
  "use strict";

  const $  = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ── media manifest (mirrors ../../assets/manifest.json) ─────── */
  const G1 = Array.from({ length: 18 }, (_, i) => {
    const id = "g1-" + String(i + 1).padStart(2, "0");
    return { id, src: `../../assets/video/g1/${id}.mp4`, poster: `../../assets/poster/${id}.jpg` };
  });

  const LOOPS = G1.slice(0, 10).map(v => ({
    id: v.id, src: `../../assets/video/loop/${v.id}.mp4`, poster: v.poster,
  }));

  const PROBE = [
    { id: "probe-01-vogue", caption: "A person is fluidly performing the vogue dance, holding a handbag and walking forward, and then stops standing idle.", rmse: "26.3 mm" },
    { id: "probe-02-walk-sway", caption: "The person is walking forward while swaying their arms.", rmse: "26.3 mm" },
    { id: "probe-03-crouch-forward", caption: "A person is crouching while moving forward, with a bent torso and legs spread wide apart.", rmse: "28.3 mm" },
    { id: "probe-04-jog-backward", caption: "A person starts jogging backward from a stationary stance.", rmse: "26.1 mm" },
    { id: "probe-05-side-steps", caption: "A person is taking several steps to the left in a hurry.", rmse: "26.3 mm" },
  ].map((p, i) => ({
    ...p,
    n: i + 1,
    src: `../../assets/video/probe/${p.id}.mp4`,
    poster: `../../assets/poster/${p.id}.jpg`,
  }));

  /* ── nav ────────────────────────────────────────────────── */
  const nav = $("#nav");
  const burger = $("#burger");
  const navLinks = $(".nav__links");
  const bar = $(".scroll-progress i");

  function onScroll() {
    const y = scrollY;
    nav.classList.toggle("is-stuck", y > 40);
    const max = document.documentElement.scrollHeight - innerHeight;
    bar.style.width = (max > 0 ? (y / max) * 100 : 0) + "%";
  }
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  burger.addEventListener("click", () => {
    const open = navLinks.classList.toggle("is-open");
    burger.setAttribute("aria-expanded", String(open));
  });
  navLinks.addEventListener("click", e => {
    if (e.target.tagName === "A") {
      navLinks.classList.remove("is-open");
      burger.setAttribute("aria-expanded", "false");
    }
  });

  // active section highlight
  const sections = $$("section[id]");
  const linkFor = {};
  $$(".nav__links a").forEach(a => { linkFor[a.getAttribute("href").slice(1)] = a; });
  const activeObs = new IntersectionObserver(entries => {
    entries.forEach(en => {
      if (!en.isIntersecting) return;
      const a = linkFor[en.target.id];
      if (!a) return;
      Object.values(linkFor).forEach(l => l.classList.remove("is-active"));
      a.classList.add("is-active");
    });
  }, { rootMargin: "-45% 0px -50% 0px" });
  sections.forEach(s => activeObs.observe(s));

  /* ── reveal on scroll ───────────────────────────────────── */
  const revealObs = new IntersectionObserver((entries, obs) => {
    entries.forEach(en => {
      if (en.isIntersecting) {
        en.target.classList.add("is-in");
        obs.unobserve(en.target);
      }
    });
  }, { rootMargin: "0px 0px -8% 0px", threshold: 0.06 });
  $$(".reveal").forEach(el => revealObs.observe(el));

  /* ── animated counters ──────────────────────────────────── */
  const countObs = new IntersectionObserver((entries, obs) => {
    entries.forEach(en => {
      if (!en.isIntersecting) return;
      obs.unobserve(en.target);
      const el = en.target;
      const target = parseFloat(el.dataset.count);
      const dec = el.dataset.fmt === "dec";
      if (reduced) {
        el.textContent = dec ? target.toFixed(2) : target.toLocaleString("en-US");
        return;
      }
      const dur = 1500, t0 = performance.now();
      (function tick(now) {
        const p = Math.min((now - t0) / dur, 1);
        const e = 1 - Math.pow(1 - p, 3);
        const v = target * e;
        el.textContent = dec ? v.toFixed(2) : Math.round(v).toLocaleString("en-US");
        if (p < 1) requestAnimationFrame(tick);
      })(t0);
    });
  }, { threshold: 0.4 });
  $$("[data-count]").forEach(el => countObs.observe(el));

  /* ── packet visualisation (1 semantic + 15 kinematic) ───── */
  const packetViz = $("#packetViz");
  if (packetViz) {
    const heights = [100, 88, 80, 74, 68, 63, 58, 54, 50, 46, 43, 40, 37, 34, 31, 28];
    heights.forEach((h, i) => {
      const cell = document.createElement("span");
      cell.className = "cell" + (i === 0 ? " cell--sem" : "");
      cell.style.height = h + "%";
      cell.title = i === 0 ? "q_sem" : `q_kin,${i}`;
      packetViz.appendChild(cell);
    });
    if (!reduced) {
      const cells = $$(".cell", packetViz);
      let k = 0;
      setInterval(() => {
        cells.forEach(c => (c.style.filter = ""));
        const c = cells[k % cells.length];
        c.style.filter = "brightness(1.55)";
        k++;
      }, 260);
    }
  }

  /* ── hero token field (canvas) ──────────────────────────── */
  const canvas = $("#tokenField");
  if (canvas && !reduced) {
    const ctx = canvas.getContext("2d");
    let W = 0, H = 0, dpr = 1, cols = [], raf = 0;
    const CW = 13, CH = 9, GAP = 5, ROWS = 16;

    function build() {
      dpr = Math.min(devicePixelRatio || 1, 2);
      W = canvas.clientWidth; H = canvas.clientHeight;
      canvas.width = W * dpr; canvas.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const colW = CW + GAP;
      const n = Math.ceil(W / colW) + 2;
      cols = Array.from({ length: n }, (_, i) => ({
        x: i * colW,
        speed: 0.14 + Math.random() * 0.1,
        depth: 1 + Math.floor(Math.random() * ROWS),
        phase: Math.random() * Math.PI * 2,
      }));
    }

    function draw(t) {
      ctx.clearRect(0, 0, W, H);
      const floor = H - 28;              // columns stack upward from here
      const colW = CW + GAP;
      const total = cols.length * colW;

      for (const c of cols) {
        c.x -= c.speed;
        if (c.x < -colW) { c.x += total; c.depth = 1 + Math.floor(Math.random() * ROWS); }

        const wob = Math.sin(t / 2600 + c.phase) * 9;
        for (let r = 0; r < c.depth; r++) {
          // r counts up from the floor; the top cell of a column is semantic
          const y = floor - (r + 1) * (CH + 4) + wob;
          const isSem = r === c.depth - 1;
          const f = r / ROWS;
          if (isSem) {
            ctx.fillStyle = `rgba(226, 72, 156, ${0.45 + 0.3 * Math.sin(t / 900 + c.phase)})`;
          } else {
            const g = 0.16 + 0.22 * f;
            ctx.fillStyle = `rgba(${Math.round(110 + 40 * f)}, ${Math.round(120 + 25 * f)}, 236, ${g})`;
          }
          ctx.beginPath();
          ctx.roundRect(c.x, y, CW, CH, 2.5);
          ctx.fill();
        }
      }
      raf = requestAnimationFrame(draw);
    }

    const heroEl = $(".hero");
    const heroObs = new IntersectionObserver(([en]) => {
      if (en.isIntersecting) { if (!raf) raf = requestAnimationFrame(draw); }
      else { cancelAnimationFrame(raf); raf = 0; }
    }, { threshold: 0 });

    if (ctx.roundRect || CanvasRenderingContext2D.prototype.roundRect) {
      build();
      heroObs.observe(heroEl);
      let rt;
      addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(build, 200); });
    }
  }

  /* ── lazy autoplay for looping videos ───────────────────── */
  const playObs = new IntersectionObserver(entries => {
    entries.forEach(en => {
      const v = en.target;
      if (en.isIntersecting) {
        if (!v.src && v.dataset.src) v.src = v.dataset.src;
        const p = v.play();
        if (p) p.then(() => {
          const card = v.closest(".vcard");
          if (card) card.classList.add("is-playing");
        }).catch(() => {});
      } else {
        v.pause();
      }
    });
  }, { rootMargin: "180px 0px", threshold: 0.18 });

  function lazyVideo(src, poster, cls) {
    const v = document.createElement("video");
    v.muted = true; v.loop = true; v.playsInline = true;
    v.setAttribute("muted", ""); v.setAttribute("playsinline", "");
    v.preload = "none";
    v.poster = poster;
    v.dataset.src = src;
    if (cls) v.className = cls;
    playObs.observe(v);
    return v;
  }

  /* ── hero film strip ────────────────────────────────────── */
  const stripTrack = $("#stripTrack");
  if (stripTrack) {
    // duplicated once so the marquee loops seamlessly at -50%
    [...LOOPS, ...LOOPS].forEach((l, i) => {
      const cell = document.createElement("div");
      cell.className = "strip__cell";
      cell.appendChild(lazyVideo(l.src, l.poster));
      cell.setAttribute("aria-hidden", i >= LOOPS.length ? "true" : "false");
      stripTrack.appendChild(cell);
    });
  }

  /* ── galleries ──────────────────────────────────────────── */
  const EXPAND_ICON =
    '<svg viewBox="0 0 24 24"><path d="M4 9V4h5M20 15v5h-5M20 9V4h-5M4 15v5h5"/></svg>';

  function card(item, kind) {
    const el = document.createElement("article");
    el.className = `vcard vcard--${kind}`;
    el.tabIndex = 0;
    el.setAttribute("role", "button");

    const media = document.createElement("div");
    media.className = "vcard__media";

    const img = document.createElement("img");
    img.src = item.poster; img.alt = item.caption || item.id; img.loading = "lazy";
    media.appendChild(img);
    media.appendChild(lazyVideo(item.src, item.poster));

    const badge = document.createElement("span");
    badge.className = "vcard__badge";
    badge.textContent = kind === "probe"
      ? `probe ${String(item.n).padStart(2, "0")}`
      : `take ${item.id.split("-")[1]}`;
    media.appendChild(badge);

    const exp = document.createElement("span");
    exp.className = "vcard__expand";
    exp.innerHTML = EXPAND_ICON;
    media.appendChild(exp);

    if (kind === "probe") {
      const panels = document.createElement("div");
      panels.className = "vcard__panels";
      panels.innerHTML = "<span>SOMA-77 direct</span><span>SMPL-X body-22</span>";
      media.appendChild(panels);
    }

    el.appendChild(media);

    if (item.caption) {
      const foot = document.createElement("div");
      foot.className = "vcard__foot";
      foot.innerHTML =
        `<p class="vcard__cap">${item.caption}</p>` +
        `<p class="vcard__meta"><span>SMPL-X fit RMSE <b>${item.rmse}</b></span>` +
        `<span>fingers driven · hands static</span></p>`;
      el.appendChild(foot);
    }
    return el;
  }

  const probeGrid = $("#probeGrid");
  const g1Grid = $("#g1Grid");

  PROBE.forEach((p, i) => {
    const el = card(p, "probe");
    el.classList.add("reveal");
    el.style.setProperty("--d", (i % 2) * 80 + "ms");
    el.addEventListener("click", () => openLB("probe", i));
    el.addEventListener("keydown", e => { if (e.key === "Enter") openLB("probe", i); });
    probeGrid.appendChild(el);
    revealObs.observe(el);
  });

  // the first clip drives the feature player, the rest fill the grid
  const GRID_G1 = G1.slice(1);
  GRID_G1.forEach((g, i) => {
    const el = card(g, "g1");
    el.classList.add("reveal");
    el.style.setProperty("--d", (i % 3) * 70 + "ms");
    el.addEventListener("click", () => openLB("g1", i + 1));
    el.addEventListener("keydown", e => { if (e.key === "Enter") openLB("g1", i + 1); });
    g1Grid.appendChild(el);
    revealObs.observe(el);
  });

  /* ── feature player + rail ──────────────────────────────── */
  const featureVideo = $("#featureVideo");
  const featureRail = $("#featureRail");
  let featureIdx = 0;

  if (featureVideo && featureRail) {
    G1.forEach((g, i) => {
      const b = document.createElement("button");
      b.className = "rail__item" + (i === 0 ? " is-active" : "");
      b.setAttribute("aria-label", `Play take ${i + 1}`);
      b.innerHTML = `<img src="${g.poster}" alt="" loading="lazy">`;
      b.addEventListener("click", () => setFeature(i));
      featureRail.appendChild(b);
    });

    const featureNow = $("#featureNow");

    function setFeature(i) {
      featureIdx = i;
      const g = G1[i];
      featureVideo.poster = g.poster;
      featureVideo.src = g.src;
      featureVideo.play().catch(() => {});
      if (featureNow) {
        featureNow.innerHTML =
          `<b>Take ${String(i + 1).padStart(2, "0")}</b><span>of ${G1.length}</span>`;
      }
      $$(".rail__item", featureRail).forEach((r, k) => r.classList.toggle("is-active", k === i));
      const active = featureRail.children[i];
      if (active) active.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
    }

    featureVideo.addEventListener("ended", () => setFeature((featureIdx + 1) % G1.length));

    // only load once the player scrolls into view
    new IntersectionObserver((entries, obs) => {
      entries.forEach(en => {
        if (!en.isIntersecting) { featureVideo.pause(); return; }
        if (!featureVideo.src) { featureVideo.src = G1[0].src; }
        featureVideo.play().catch(() => {});
      });
    }, { threshold: 0.25 }).observe(featureVideo);

    $("[data-expand]").addEventListener("click", () => openLB("g1", featureIdx));
  }

  /* ── lightbox ───────────────────────────────────────────── */
  const lb = $("#lb");
  const lbStage = $("#lbStage");
  const lbCap = $("#lbCap");
  let lbList = [], lbIdx = 0, lastFocus = null;

  const FIGS = $$("[data-zoom]").map(a => ({
    type: "img", src: a.getAttribute("href"), caption: a.dataset.cap,
  }));
  $$("[data-zoom]").forEach((a, i) => {
    a.addEventListener("click", e => { e.preventDefault(); openLB("fig", i); });
  });

  function listFor(kind) {
    if (kind === "probe") {
      return PROBE.map(p => ({ type: "video", src: p.src, poster: p.poster, caption: p.caption }));
    }
    if (kind === "g1") {
      return G1.map((g, i) => ({
        type: "video", src: g.src, poster: g.poster,
        caption: `Unitree G1 · take ${String(i + 1).padStart(2, "0")} — motion tracked on hardware through a SONIC whole-body controller.`,
      }));
    }
    return FIGS;
  }

  function openLB(kind, idx) {
    lbList = listFor(kind);
    lbIdx = idx;
    lastFocus = document.activeElement;
    lb.classList.add("is-open");
    lb.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
    requestAnimationFrame(() => lb.classList.add("is-visible"));
    renderLB();
    $("[data-lb-close]").focus();
  }

  function closeLB() {
    lb.classList.remove("is-visible");
    lb.setAttribute("aria-hidden", "true");
    document.body.style.overflow = "";
    setTimeout(() => { lb.classList.remove("is-open"); lbStage.innerHTML = ""; }, 320);
    if (lastFocus) lastFocus.focus();
  }

  function renderLB() {
    const it = lbList[lbIdx];
    lbStage.innerHTML = "";
    if (it.type === "video") {
      const v = document.createElement("video");
      v.src = it.src; v.poster = it.poster;
      v.controls = true; v.autoplay = true; v.loop = true;
      v.muted = true; v.playsInline = true;
      lbStage.appendChild(v);
      v.play().catch(() => {});
    } else {
      const img = document.createElement("img");
      img.src = it.src; img.alt = it.caption || "";
      lbStage.appendChild(img);
    }
    lbCap.textContent = it.caption || "";
    const single = lbList.length < 2;
    $("[data-lb-prev]").classList.toggle("is-hidden", single);
    $("[data-lb-next]").classList.toggle("is-hidden", single);
  }

  function step(d) {
    lbIdx = (lbIdx + d + lbList.length) % lbList.length;
    renderLB();
  }

  $("[data-lb-close]").addEventListener("click", closeLB);
  $("[data-lb-prev]").addEventListener("click", () => step(-1));
  $("[data-lb-next]").addEventListener("click", () => step(1));
  lb.addEventListener("click", e => { if (e.target === lb || e.target === lbStage) closeLB(); });
  addEventListener("keydown", e => {
    if (!lb.classList.contains("is-open")) return;
    if (e.key === "Escape") closeLB();
    if (e.key === "ArrowLeft") step(-1);
    if (e.key === "ArrowRight") step(1);
  });

  /* ── bibtex copy ────────────────────────────────────────── */
  const bibBtn = $("#bibCopy");
  if (bibBtn) {
    bibBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText($("#bibText").textContent);
        bibBtn.textContent = "Copied";
      } catch {
        bibBtn.textContent = "Press ⌘C";
      }
      setTimeout(() => (bibBtn.textContent = "Copy"), 1800);
    });
  }
})();

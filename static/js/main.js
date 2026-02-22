/**
 * FreshScan AI — Frontend Logic
 * Handles drag-and-drop, upload, API call, and result rendering.
 */

(() => {
    "use strict";

    // ── DOM refs ──────────────────────────────
    const uploadZone = document.getElementById("uploadZone");
    const fileInput = document.getElementById("fileInput");
    const previewArea = document.getElementById("previewArea");
    const previewImage = document.getElementById("previewImage");
    const removeBtn = document.getElementById("removeBtn");
    const analyseBtn = document.getElementById("analyseBtn");
    const btnLoader = document.getElementById("btnLoader");
    const btnText = document.getElementById("btnText");

    const resultPlaceholder = document.getElementById("resultPlaceholder");
    const resultContent = document.getElementById("resultContent");
    const verdictIcon = document.getElementById("verdictIcon");
    const verdictLabel = document.getElementById("verdictLabel");
    const verdictMethod = document.getElementById("verdictMethod");
    const ringFill = document.getElementById("ringFill");
    const confidenceValue = document.getElementById("confidenceValue");
    const scoreBreakdown = document.getElementById("scoreBreakdown");
    const colorBar = document.getElementById("colorBar");
    const textureBar = document.getElementById("textureBar");
    const edgeBar = document.getElementById("edgeBar");
    const colorVal = document.getElementById("colorVal");
    const textureVal = document.getElementById("textureVal");
    const edgeVal = document.getElementById("edgeVal");

    let selectedFile = null;

    // ── Drag & Drop ──────────────────────────
    uploadZone.addEventListener("click", () => fileInput.click());

    uploadZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadZone.classList.add("drag-over");
    });
    uploadZone.addEventListener("dragleave", () => {
        uploadZone.classList.remove("drag-over");
    });
    uploadZone.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadZone.classList.remove("drag-over");
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith("image/")) handleFile(file);
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length) handleFile(fileInput.files[0]);
    });

    // ── File handling ────────────────────────
    function handleFile(file) {
        selectedFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadZone.style.display = "none";
            previewArea.style.display = "block";
            analyseBtn.disabled = false;
        };
        reader.readAsDataURL(file);

        // Reset results
        resetResult();
    }

    removeBtn.addEventListener("click", () => {
        selectedFile = null;
        fileInput.value = "";
        previewArea.style.display = "none";
        uploadZone.style.display = "block";
        analyseBtn.disabled = true;
        resetResult();
    });

    // ── Analyse button ───────────────────────
    analyseBtn.addEventListener("click", async () => {
        if (!selectedFile) return;

        // UI → loading
        analyseBtn.disabled = true;
        btnLoader.style.display = "inline-block";
        btnText.textContent = "Analysing…";
        resetResult();

        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            const res = await fetch("/predict", { method: "POST", body: formData });
            const data = await res.json();

            if (data.error) {
                alert("Error: " + data.error);
            } else {
                renderResult(data);
            }
        } catch (err) {
            alert("Network error — is the server running?");
            console.error(err);
        } finally {
            analyseBtn.disabled = false;
            btnLoader.style.display = "none";
            btnText.textContent = "Analyse Freshness";
        }
    });

    // ── Render result ────────────────────────
    function renderResult(data) {
        resultPlaceholder.style.display = "none";
        resultContent.style.display = "block";

        const isFresh = data.label === "Fresh";
        const color = isFresh ? "var(--fresh-green)" : "var(--expired-red)";

        // Verdict
        verdictIcon.textContent = isFresh ? "✅" : "⛔";
        verdictLabel.textContent = data.label;
        verdictLabel.className = "verdict-label " + (isFresh ? "fresh" : "expired");
        verdictMethod.textContent = "Method: " + data.method;

        // Confidence ring
        const circumference = 2 * Math.PI * 52; // r=52
        const offset = circumference * (1 - data.confidence / 100);
        ringFill.style.stroke = color;
        // Use requestAnimationFrame to trigger animation
        requestAnimationFrame(() => {
            ringFill.style.strokeDashoffset = offset;
        });
        animateValue(confidenceValue, 0, data.confidence, 1000);

        // Score bars
        const d = data.details || {};
        if (d.color_score !== null && d.color_score !== undefined) {
            scoreBreakdown.style.display = "block";
            animateBar(colorBar, d.color_score, color);
            colorVal.textContent = d.color_score + "%";

            animateBar(textureBar, d.texture_score, color);
            textureVal.textContent = d.texture_score + "%";

            animateBar(edgeBar, d.edge_score, color);
            edgeVal.textContent = d.edge_score + "%";
        } else {
            // CNN mode — no per-category breakdown
            scoreBreakdown.style.display = "none";
        }
    }

    function resetResult() {
        resultPlaceholder.style.display = "block";
        resultContent.style.display = "none";
        ringFill.style.strokeDashoffset = 326.73;
        confidenceValue.textContent = "0%";
        colorBar.style.width = "0%";
        textureBar.style.width = "0%";
        edgeBar.style.width = "0%";
        colorVal.textContent = "—";
        textureVal.textContent = "—";
        edgeVal.textContent = "—";
    }

    // ── Helpers ──────────────────────────────
    function animateBar(el, value, color) {
        el.style.background = color;
        requestAnimationFrame(() => {
            el.style.width = value + "%";
        });
    }

    function animateValue(el, start, end, duration) {
        const startTime = performance.now();
        function step(now) {
            const progress = Math.min((now - startTime) / duration, 1);
            const current = (start + (end - start) * easeOut(progress)).toFixed(1);
            el.textContent = current + "%";
            if (progress < 1) requestAnimationFrame(step);
        }
        requestAnimationFrame(step);
    }

    function easeOut(t) {
        return 1 - Math.pow(1 - t, 3);
    }

    // ── Navbar scroll effect ─────────────────
    window.addEventListener("scroll", () => {
        const nav = document.getElementById("navbar");
        if (window.scrollY > 60) {
            nav.style.background = "rgba(10, 14, 23, 0.95)";
        } else {
            nav.style.background = "rgba(10, 14, 23, 0.75)";
        }
    });
})();

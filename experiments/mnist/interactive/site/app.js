const payloadPath = "./data/posterior.json";

const elements = {
  heroCopy: document.getElementById("hero-copy"),
  plotTitle: document.getElementById("plot-title"),
  plotDescription: document.getElementById("plot-description"),
  selectionSummary: document.getElementById("selection-summary"),
  runSummary: document.getElementById("run-summary"),
  sourceRun: document.getElementById("source-run"),
  legend: document.getElementById("legend"),
  resetButton: document.getElementById("reset-view"),
  plot: document.getElementById("plot"),
  hoverCard: document.getElementById("hover-card"),
  hoverImage: document.getElementById("hover-image"),
  hoverLabel: document.getElementById("hover-label"),
  hoverIndex: document.getElementById("hover-index"),
  hoverCoords: document.getElementById("hover-coords"),
};

let activePayload = null;
let pendingCamera = null;
let depthFrame = null;
let lastMousePosition = { x: window.innerWidth / 2, y: window.innerHeight / 2 };

function formatValue(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return String(value);
  }
  if (Math.abs(value) >= 100) {
    return value.toFixed(2);
  }
  if (Math.abs(value) >= 10) {
    return value.toFixed(3);
  }
  return value.toFixed(4);
}

function sphereSurfaceTrace() {
  const u = [];
  const v = [];
  for (let i = 0; i <= 36; i += 1) {
    u.push((i / 36) * Math.PI * 2);
  }
  for (let i = 0; i <= 18; i += 1) {
    v.push((i / 18) * Math.PI);
  }
  const x = v.map((phi) => u.map((theta) => Math.cos(theta) * Math.sin(phi)));
  const y = v.map((phi) => u.map((theta) => Math.sin(theta) * Math.sin(phi)));
  const z = v.map((phi) => u.map(() => Math.cos(phi)));
  return {
    type: "surface",
    x,
    y,
    z,
    showscale: false,
    hoverinfo: "skip",
    opacity: 0.22,
    colorscale: [
      [0, "#1e3658"],
      [1, "#2a4a76"],
    ],
    contours: {
      x: { show: false },
      y: { show: false },
      z: { show: false },
    },
  };
}

function sphereWireframeTraces() {
  const traces = [];
  const latitudes = [-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75];
  const longitudes = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330];

  latitudes.forEach((degrees) => {
    const phi = (degrees * Math.PI) / 180;
    const x = [];
    const y = [];
    const z = [];
    for (let step = 0; step <= 80; step += 1) {
      const theta = (step / 80) * Math.PI * 2;
      x.push(Math.cos(theta) * Math.cos(phi));
      y.push(Math.sin(theta) * Math.cos(phi));
      z.push(Math.sin(phi));
    }
    traces.push({
      type: "scatter3d",
      mode: "lines",
      x,
      y,
      z,
      hoverinfo: "skip",
      showlegend: false,
      line: { color: "rgba(173, 219, 255, 0.22)", width: 2.2 },
    });
  });

  longitudes.forEach((degrees) => {
    const theta = (degrees * Math.PI) / 180;
    const x = [];
    const y = [];
    const z = [];
    for (let step = 0; step <= 80; step += 1) {
      const phi = -Math.PI / 2 + (step / 80) * Math.PI;
      x.push(Math.cos(theta) * Math.cos(phi));
      y.push(Math.sin(theta) * Math.cos(phi));
      z.push(Math.sin(phi));
    }
    traces.push({
      type: "scatter3d",
      mode: "lines",
      x,
      y,
      z,
      hoverinfo: "skip",
      showlegend: false,
      line: { color: "rgba(173, 219, 255, 0.22)", width: 2.2 },
    });
  });

  return traces;
}

function anchorTrace() {
  const coordinates = [-1.05, 1.05];
  const x = [];
  const y = [];
  const z = [];
  coordinates.forEach((cx) => {
    coordinates.forEach((cy) => {
      coordinates.forEach((cz) => {
        x.push(cx);
        y.push(cy);
        z.push(cz);
      });
    });
  });
  return {
    type: "scatter3d",
    mode: "markers",
    x,
    y,
    z,
    hoverinfo: "skip",
    showlegend: false,
    marker: {
      size: 1,
      color: "rgba(255,255,255,0.001)",
      opacity: 0.001,
    },
  };
}

function pointTraces(payload) {
  return payload.label_names.map((labelName, label) => {
    const points = payload.points.filter((point) => point.label === label);
    return {
      type: "scatter3d",
      mode: "markers",
      name: labelName,
      x: points.map((point) => point.x),
      y: points.map((point) => point.y),
      z: points.map((point) => point.z),
      customdata: points.map((point) => [
        point.subset_index,
        point.test_index,
        point.image_uri,
        point.label_name,
      ]),
      hovertemplate: "<extra></extra>",
      showlegend: false,
      marker: {
        size: 6.2,
        opacity: 0.98,
        color: payload.palette[label % payload.palette.length],
        line: { width: 1.1, color: "rgba(8, 12, 22, 0.88)" },
      },
    };
  });
}

function normalizeVector(vector) {
  const norm = Math.hypot(vector.x, vector.y, vector.z);
  if (norm < 1e-8) {
    return { x: 1, y: 1, z: 1 };
  }
  return {
    x: vector.x / norm,
    y: vector.y / norm,
    z: vector.z / norm,
  };
}

function depthOpacityValue(dot) {
  const clamped = Math.max(-1, Math.min(1, dot));
  const shifted = (clamped + 0.08) / 0.24;
  const t = Math.max(0, Math.min(1, shifted));
  const smooth = t * t * (3 - 2 * t);
  return 0.02 + 0.96 * smooth;
}

function hexToRgb(hex) {
  const normalized = hex.replace("#", "");
  const value = normalized.length === 3
    ? normalized.split("").map((character) => character + character).join("")
    : normalized;
  const integer = Number.parseInt(value, 16);
  return {
    r: (integer >> 16) & 255,
    g: (integer >> 8) & 255,
    b: integer & 255,
  };
}

function applyDepthOpacity(camera = null) {
  const defaultEye = activePayload?.default_camera?.eye ?? { x: 1, y: 1, z: 1 };
  const eye = normalizeVector(camera?.eye ?? defaultEye);
  const updateIndices = [];
  const colorArrays = [];
  const sizeArrays = [];

  elements.plot.data.forEach((trace, index) => {
    if (trace.type !== "scatter3d" || trace.mode !== "markers" || !trace.customdata) {
      return;
    }
    const labelIndex = activePayload.label_names.indexOf(trace.name);
    const rgb = hexToRgb(activePayload.palette[labelIndex % activePayload.palette.length]);
    const colors = trace.x.map((x, pointIndex) => {
      const dot = x * eye.x + trace.y[pointIndex] * eye.y + trace.z[pointIndex] * eye.z;
      const alpha = depthOpacityValue(dot);
      return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha.toFixed(3)})`;
    });
    const sizes = trace.x.map((x, pointIndex) => {
      const dot = x * eye.x + trace.y[pointIndex] * eye.y + trace.z[pointIndex] * eye.z;
      return 4.9 + 4.9 * depthOpacityValue(dot);
    });
    updateIndices.push(index);
    colorArrays.push(colors);
    sizeArrays.push(sizes);
  });

  if (updateIndices.length > 0) {
    Plotly.restyle(
      elements.plot,
      { "marker.color": colorArrays, "marker.size": sizeArrays },
      updateIndices,
    );
  }
}

function scheduleDepthOpacity(camera) {
  pendingCamera = camera;
  if (depthFrame !== null) {
    return;
  }
  depthFrame = window.requestAnimationFrame(() => {
    applyDepthOpacity(pendingCamera);
    pendingCamera = null;
    depthFrame = null;
  });
}

function plotLayout(payload) {
  return {
    margin: { l: 0, r: 0, t: 0, b: 0 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    showlegend: false,
    scene: {
      aspectmode: "manual",
      aspectratio: { x: 1, y: 1, z: 1 },
      dragmode: "turntable",
      camera: payload.default_camera,
      xaxis: { visible: false, range: [-1.08, 1.08], autorange: false },
      yaxis: { visible: false, range: [-1.08, 1.08], autorange: false },
      zaxis: { visible: false, range: [-1.08, 1.08], autorange: false },
      bgcolor: "rgba(0,0,0,0)",
    },
  };
}

function renderLegend(payload) {
  elements.legend.replaceChildren();
  payload.label_names.forEach((labelName, label) => {
    const chip = document.createElement("div");
    chip.className = "legend-chip";
    const swatch = document.createElement("span");
    swatch.className = "legend-swatch";
    swatch.style.color = payload.palette[label % payload.palette.length];
    swatch.style.backgroundColor = payload.palette[label % payload.palette.length];
    const text = document.createElement("span");
    text.textContent = labelName;
    chip.append(swatch, text);
    elements.legend.appendChild(chip);
  });
}

function metaChip(label, value) {
  const chip = document.createElement("div");
  chip.className = "meta-chip";
  const chipLabel = document.createElement("p");
  chipLabel.className = "meta-chip-label";
  chipLabel.textContent = label;
  const chipValue = document.createElement("p");
  chipValue.className = "meta-chip-value";
  chipValue.textContent = value;
  chip.append(chipLabel, chipValue);
  return chip;
}

function runLine(label, value) {
  const line = document.createElement("p");
  const strong = document.createElement("strong");
  strong.textContent = `${label}: `;
  line.append(strong, document.createTextNode(value));
  return line;
}

function renderMeta(payload) {
  elements.heroCopy.textContent = "Rotate the selected paper run's posterior means and hover over any point to inspect the original MNIST digit and its position on the sphere.";
  elements.plotTitle.textContent = payload.title;
  elements.plotDescription.textContent = payload.description;
  elements.selectionSummary.replaceChildren(
    metaChip("Seed", String(payload.model.seed)),
    metaChip("Selected epoch", String(payload.model.epoch)),
    metaChip("Eval recon", formatValue(payload.metrics.reconstruction_loss)),
    metaChip("Eval total", formatValue(payload.metrics.total_loss)),
    metaChip("Eval KL", formatValue(payload.metrics.kl)),
    metaChip("Points", String(payload.num_points)),
  );
  elements.runSummary.replaceChildren(
    runLine("Model", "spherical Cauchy"),
    runLine("Latent dimension", `${payload.model.intrinsic_dimension} intrinsic / ${payload.model.ambient_dimension} ambient`),
    runLine("KL route", payload.model.kl_method),
    runLine("Selection", payload.selection.rule),
  );
  elements.sourceRun.textContent = `Source figure: ${payload.paper_figure.path} · input SHA256: ${payload.input_sha256}`;
}

function hideHoverCard() {
  elements.hoverCard.style.display = "none";
  elements.hoverCard.setAttribute("aria-hidden", "true");
}

function showHoverCard(point, mouseEvent = null) {
  const [subsetIndex, testIndex, imageUri, labelName] = point.customdata;
  elements.hoverImage.src = imageUri;
  elements.hoverLabel.textContent = `Digit ${labelName}`;
  elements.hoverIndex.textContent = `point ${subsetIndex} · test idx ${testIndex}`;
  elements.hoverCoords.textContent = `(${point.x.toFixed(3)}, ${point.y.toFixed(3)}, ${point.z.toFixed(3)})`;

  const offset = 18;
  elements.hoverCard.style.display = "block";
  elements.hoverCard.setAttribute("aria-hidden", "false");
  const pointerX = mouseEvent?.clientX ?? lastMousePosition.x;
  const pointerY = mouseEvent?.clientY ?? lastMousePosition.y;
  const left = Math.min(pointerX + offset, window.innerWidth - 208);
  const top = Math.min(pointerY + offset, window.innerHeight - 262);
  elements.hoverCard.style.left = `${Math.max(12, left)}px`;
  elements.hoverCard.style.top = `${Math.max(12, top)}px`;
}

async function loadPayload() {
  const response = await fetch(payloadPath);
  if (!response.ok) {
    throw new Error(`Failed to load ${payloadPath}`);
  }
  return response.json();
}

async function renderPlot(payload) {
  const traces = [
    anchorTrace(),
    sphereSurfaceTrace(),
    ...sphereWireframeTraces(),
    ...pointTraces(payload),
  ];
  await Plotly.newPlot(elements.plot, traces, plotLayout(payload), {
    responsive: true,
    scrollZoom: true,
    displaylogo: false,
    displayModeBar: true,
    modeBarButtonsToRemove: ["lasso2d", "select2d", "zoom2d", "pan2d", "toImage"],
  });
  applyDepthOpacity(payload.default_camera);

  elements.plot.on("plotly_hover", (event) => {
    const point = event.points?.[0];
    if (point?.customdata) {
      showHoverCard(point, event.event ?? null);
    }
  });
  elements.plot.on("plotly_unhover", hideHoverCard);
  elements.plot.on("plotly_relayout", (event) => {
    if (event["scene.camera"]) {
      scheduleDepthOpacity(event["scene.camera"]);
    }
  });
  elements.resetButton.addEventListener("click", () => {
    Plotly.relayout(elements.plot, { "scene.camera": payload.default_camera });
    scheduleDepthOpacity(payload.default_camera);
  });
  window.addEventListener("resize", () => Plotly.Plots.resize(elements.plot));
  window.addEventListener("mousemove", (event) => {
    lastMousePosition = { x: event.clientX, y: event.clientY };
  });
}

async function init() {
  activePayload = await loadPayload();
  renderLegend(activePayload);
  renderMeta(activePayload);
  await renderPlot(activePayload);
}

init().catch((error) => {
  elements.plotTitle.textContent = "Failed to load interactive sphere";
  elements.plotDescription.textContent = error.message;
  elements.plot.textContent = error.message;
  elements.plot.classList.add("plot-error");
  console.error(error);
});

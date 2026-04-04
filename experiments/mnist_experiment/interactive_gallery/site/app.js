const payloadPath = "./data/mnist_spcauchy_s2.json";

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

  latitudes.forEach((deg) => {
    const phi = (deg * Math.PI) / 180;
    const xs = [];
    const ys = [];
    const zs = [];
    for (let step = 0; step <= 80; step += 1) {
      const theta = (step / 80) * Math.PI * 2;
      xs.push(Math.cos(theta) * Math.cos(phi));
      ys.push(Math.sin(theta) * Math.cos(phi));
      zs.push(Math.sin(phi));
    }
    traces.push({
      type: "scatter3d",
      mode: "lines",
      x: xs,
      y: ys,
      z: zs,
      hoverinfo: "skip",
      showlegend: false,
      line: { color: "rgba(173, 219, 255, 0.22)", width: 2.2 },
    });
  });

  longitudes.forEach((deg) => {
    const theta = (deg * Math.PI) / 180;
    const xs = [];
    const ys = [];
    const zs = [];
    for (let step = 0; step <= 80; step += 1) {
      const phi = -Math.PI / 2 + (step / 80) * Math.PI;
      xs.push(Math.cos(theta) * Math.cos(phi));
      ys.push(Math.sin(theta) * Math.cos(phi));
      zs.push(Math.sin(phi));
    }
    traces.push({
      type: "scatter3d",
      mode: "lines",
      x: xs,
      y: ys,
      z: zs,
      hoverinfo: "skip",
      showlegend: false,
      line: { color: "rgba(173, 219, 255, 0.22)", width: 2.2 },
    });
  });

  return traces;
}

function anchorTrace() {
  const coords = [-1.05, 1.05];
  const x = [];
  const y = [];
  const z = [];
  coords.forEach((cx) => {
    coords.forEach((cy) => {
      coords.forEach((cz) => {
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
    const xs = [];
    const ys = [];
    const zs = [];
    const customdata = [];
    payload.points.forEach((point) => {
      if (point.label !== label) {
        return;
      }
      xs.push(point.x);
      ys.push(point.y);
      zs.push(point.z);
      customdata.push([
        point.subset_index,
        point.dataset_index,
        point.image_uri,
        point.label_name,
      ]);
    });
    return {
      type: "scatter3d",
      mode: "markers",
      name: labelName,
      x: xs,
      y: ys,
      z: zs,
      customdata,
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
    ? normalized.split("").map((char) => char + char).join("")
    : normalized;
  const intValue = Number.parseInt(value, 16);
  return {
    r: (intValue >> 16) & 255,
    g: (intValue >> 8) & 255,
    b: intValue & 255,
  };
}

function applyDepthOpacity(camera = null) {
  const eye = normalizeVector(camera?.eye ?? activePayload?.default_camera?.eye ?? { x: 1, y: 1, z: 1 });
  const updateIndices = [];
  const colorArrays = [];
  const sizeArrays = [];

  elements.plot.data.forEach((trace, index) => {
    if (trace.type !== "scatter3d" || trace.mode !== "markers" || !trace.customdata) {
      return;
    }
    const labelName = trace.name;
    const labelIndex = activePayload.label_names.indexOf(labelName);
    const rgb = hexToRgb(activePayload.palette[labelIndex % activePayload.palette.length]);
    const colors = trace.x.map((x, pointIndex) => {
      const y = trace.y[pointIndex];
      const z = trace.z[pointIndex];
      const dot = x * eye.x + y * eye.y + z * eye.z;
      const alpha = depthOpacityValue(dot);
      return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha.toFixed(3)})`;
    });
    const sizes = trace.x.map((x, pointIndex) => {
      const y = trace.y[pointIndex];
      const z = trace.z[pointIndex];
      const dot = x * eye.x + y * eye.y + z * eye.z;
      const alpha = depthOpacityValue(dot);
      return 4.9 + 4.9 * alpha;
    });
    updateIndices.push(index);
    colorArrays.push(colors);
    sizeArrays.push(sizes);
  });

  if (updateIndices.length === 0) {
    return;
  }

  Plotly.restyle(
    elements.plot,
    {
      "marker.color": colorArrays,
      "marker.size": sizeArrays,
    },
    updateIndices,
  );
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

function renderMeta(payload) {
  elements.heroCopy.textContent = "Rotatable posterior means from the qualitative MNIST spCauchy-VAE run, with hoverable original digits for local inspection of class structure on the sphere.";
  elements.plotTitle.textContent = payload.title;
  elements.plotDescription.textContent = payload.description;

  const summary = payload.selection_summary;
  elements.selectionSummary.replaceChildren(
    metaChip("Selected Epoch", String(summary.selected_epoch)),
    metaChip("Eval Recon", formatValue(summary.selected_eval_recon_loss)),
    metaChip("Eval Total", formatValue(summary.selected_eval_total_loss)),
    metaChip("Eval KL", formatValue(summary.selected_eval_kl)),
    metaChip("Points", String(payload.num_points)),
    metaChip("Per Digit", String(payload.points_per_label)),
  );

  const runConfig = payload.run_config;
  elements.runSummary.innerHTML = `
    <p><strong>Model:</strong> ${runConfig.model_family}</p>
    <p><strong>Ambient latent dim:</strong> ${runConfig.ambient_latent_dim}</p>
    <p><strong>Epochs:</strong> ${runConfig.epochs}</p>
    <p><strong>Learning rate:</strong> ${runConfig.learning_rate}</p>
    <p><strong>Seed:</strong> ${runConfig.seed}</p>
  `;
  elements.sourceRun.textContent = payload.source_run;
}

function hideHoverCard() {
  elements.hoverCard.style.display = "none";
  elements.hoverCard.setAttribute("aria-hidden", "true");
}

function showHoverCard(point, mouseEvent = null) {
  const [subsetIndex, datasetIndex, imageUri, labelName] = point.customdata;
  elements.hoverImage.src = imageUri;
  elements.hoverLabel.textContent = `Digit ${labelName}`;
  elements.hoverIndex.textContent = `point ${subsetIndex} - eval idx ${datasetIndex}`;
  elements.hoverCoords.textContent = `(${point.x.toFixed(3)}, ${point.y.toFixed(3)}, ${point.z.toFixed(3)})`;

  const card = elements.hoverCard;
  const offset = 18;
  card.style.display = "block";
  card.setAttribute("aria-hidden", "false");

  const width = 196;
  const height = 250;
  const pointerX = mouseEvent?.clientX ?? lastMousePosition.x;
  const pointerY = mouseEvent?.clientY ?? lastMousePosition.y;
  const rawLeft = pointerX + offset;
  const rawTop = pointerY + offset;
  const left = Math.min(rawLeft, window.innerWidth - width - 12);
  const top = Math.min(rawTop, window.innerHeight - height - 12);
  card.style.left = `${Math.max(12, left)}px`;
  card.style.top = `${Math.max(12, top)}px`;
}

async function loadPayload() {
  const response = await fetch(payloadPath);
  if (!response.ok) {
    throw new Error(`Failed to load ${payloadPath}`);
  }
  return response.json();
}

async function renderPlot(payload) {
  const traces = [anchorTrace(), sphereSurfaceTrace(), ...sphereWireframeTraces(), ...pointTraces(payload)];
  await Plotly.newPlot(elements.plot, traces, plotLayout(payload), {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ["lasso2d", "select2d", "zoom2d", "pan2d", "toImage"],
  });
  applyDepthOpacity(payload.default_camera);

  elements.plot.on("plotly_hover", (event) => {
    if (!event.points || event.points.length === 0) {
      return;
    }
    const point = event.points[0];
    if (!point.customdata) {
      return;
    }
    showHoverCard(point, event.event ?? null);
  });

  elements.plot.on("plotly_unhover", () => {
    hideHoverCard();
  });

  elements.plot.on("plotly_relayout", (event) => {
    const camera = event["scene.camera"];
    if (camera) {
      applyDepthOpacity(camera);
    }
  });

  elements.resetButton.addEventListener("click", () => {
    Plotly.relayout(elements.plot, { "scene.camera": payload.default_camera });
    applyDepthOpacity(payload.default_camera);
  });

  window.addEventListener("resize", () => {
    Plotly.Plots.resize(elements.plot);
  });
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
  console.error(error);
});

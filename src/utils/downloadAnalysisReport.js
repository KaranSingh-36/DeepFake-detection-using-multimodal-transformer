function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => {
    const entities = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    };

    return entities[char];
  });
}

export function downloadAnalysisReport(item) {
  const isDemo = item.isDemo !== false;
  const reportStatus = isDemo
    ? "DEMO — NOT A REAL MODEL PREDICTION"
    : "MODEL RESULT — verify backend provenance";

  const reportHtml = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FusionTrace AI Analysis Report</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 24px; color: #1e293b; line-height: 1.6; }
    header { border-bottom: 3px solid #2563eb; padding-bottom: 16px; margin-bottom: 24px; }
    h1 { margin-bottom: 4px; }
    .subtitle { color: #64748b; }
    .status { padding: 14px; background: #fff7ed; color: #9a3412; border: 1px solid #fed7aa; border-radius: 8px; font-weight: bold; margin: 20px 0; }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 12px; border-bottom: 1px solid #e2e8f0; overflow-wrap: anywhere; }
    th { width: 38%; }
    .notice { margin-top: 24px; font-size: 13px; color: #64748b; }
    button { padding: 10px 16px; border: 0; border-radius: 6px; background: #2563eb; color: white; cursor: pointer; }
    @media print { button { display: none; } body { margin: 0; } }
  </style>
</head>
<body>
  <header>
    <h1>FusionTrace AI</h1>
    <div class="subtitle">Multimodal Deepfake Detection</div>
    <div class="subtitle">Analysis Report</div>
  </header>
  <div class="status">${escapeHtml(reportStatus)}</div>
  <table>
    <tr><th>Video filename</th><td>${escapeHtml(item.fileName)}</td></tr>
    <tr><th>Prediction</th><td>${escapeHtml(item.prediction)}</td></tr>
    <tr><th>Confidence</th><td>${escapeHtml(item.confidence)}%</td></tr>
    <tr><th>Processing time</th><td>${escapeHtml(item.processingTime)} seconds</td></tr>
    <tr><th>Analysis time</th><td>${escapeHtml(item.analyzedAt)}</td></tr>
    <tr><th>Report generated</th><td>${escapeHtml(new Date().toLocaleString())}</td></tr>
  </table>
  <p class="notice">${isDemo
    ? "This report contains a simulated frontend result. It is not evidence that the trained model analyzed this video. Do not use it to make authenticity or cybersecurity decisions."
    : "Verify this result against the actual model service and its recorded inference metadata before relying on it."}</p>
  <button onclick="window.print()">Print / Save as PDF</button>
</body>
</html>`;

  const blob = new Blob([reportHtml], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `FusionTrace-Report-${Date.now()}.html`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

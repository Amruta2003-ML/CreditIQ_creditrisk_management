document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("applicantForm");
  const annualIncInput = document.getElementById("annual_inc");
  const monthlyIncDisp = document.getElementById("monthly_inc_disp");
  const loanAmntInput = document.getElementById("loan_amnt");
  const intRateInput = document.getElementById("int_rate");
  const termSelect = document.getElementById("term");
  const installmentInput = document.getElementById("installment");
  const apiStatusText = document.getElementById("apiStatusText");
  const apiStatusBadge = document.getElementById("apiStatusBadge");

  // Output elements
  const gaugeFill = document.getElementById("gaugeFill");
  const pdPercentage = document.getElementById("pdPercentage");
  const riskTierText = document.getElementById("riskTierText");
  const creditScoreText = document.getElementById("creditScoreText");
  const decisionBadge = document.getElementById("decisionBadge");
  const recommendationBox = document.getElementById("recommendationBox");
  const recIcon = document.getElementById("recIcon");
  const recTitle = document.getElementById("recTitle");
  const recDesc = document.getElementById("recDesc");
  const riskDriversList = document.getElementById("riskDriversList");
  const rawJsonDisplay = document.getElementById("rawJsonDisplay");
  const submitBtn = document.getElementById("submitBtn");

  // API Base URL (Relative if served from FastAPI, or localhost default)
  const API_ENDPOINT = "/api/v1/score_applicant";
  const HEALTH_ENDPOINT = "/health";

  // Check API health
  async function checkHealth() {
    try {
      const res = await fetch(HEALTH_ENDPOINT);
      if (res.ok) {
        const data = await res.json();
        if (data.engine_loaded) {
          apiStatusText.textContent = "AI Engine Online";
          apiStatusBadge.querySelector(".status-dot").style.background = "#10b981";
        } else {
          apiStatusText.textContent = "Model Unloaded";
          apiStatusBadge.querySelector(".status-dot").style.background = "#f59e0b";
        }
      }
    } catch {
      apiStatusText.textContent = "Standby (Local Mode)";
      apiStatusBadge.querySelector(".status-dot").style.background = "#38bdf8";
    }
  }
  checkHealth();

  // Dynamic calculations: Monthly Income & Estimated Installment
  function updateFinancialEstimates() {
    const annual = parseFloat(annualIncInput.value) || 0;
    const monthly = Math.round(annual / 12);
    monthlyIncDisp.value = `$${monthly.toLocaleString()}`;

    const principal = parseFloat(loanAmntInput.value) || 0;
    const rate = (parseFloat(intRateInput.value) || 0) / 100 / 12;
    const months = termSelect.value.includes("60") ? 60 : 36;

    if (principal > 0 && rate > 0) {
      const pmt = (principal * (rate * Math.pow(1 + rate, months))) / (Math.pow(1 + rate, months) - 1);
      installmentInput.value = pmt.toFixed(2);
    }
  }

  annualIncInput.addEventListener("input", updateFinancialEstimates);
  loanAmntInput.addEventListener("input", updateFinancialEstimates);
  intRateInput.addEventListener("input", updateFinancialEstimates);
  termSelect.addEventListener("change", updateFinancialEstimates);

  // Preset Profiles
  const PRESETS = {
    prime: {
      loan_amnt: 15000,
      term: " 36 months",
      int_rate: 8.99,
      annual_inc: 85000,
      emp_length: "6 years",
      home_ownership: "MORTGAGE",
      verification_status: "Verified",
      purpose: "debt_consolidation",
      fico_range_low: 740,
      fico_range_high: 744,
      dti: 14.5,
      earliest_cr_line: "Jan-2008",
      inq_last_6mths: 0,
      delinq_2yrs: 0,
      open_acc: 12,
      total_acc: 22,
      revol_bal: 12400,
      revol_util: "28.5%",
      pub_rec: 0,
      collections_12_mths_ex_med: 0
    },
    nearPrime: {
      loan_amnt: 22000,
      term: " 36 months",
      int_rate: 14.85,
      annual_inc: 58000,
      emp_length: "3 years",
      home_ownership: "RENT",
      verification_status: "Source Verified",
      purpose: "credit_card",
      fico_range_low: 675,
      fico_range_high: 679,
      dti: 24.2,
      earliest_cr_line: "May-2013",
      inq_last_6mths: 2,
      delinq_2yrs: 1,
      open_acc: 9,
      total_acc: 16,
      revol_bal: 19500,
      revol_util: "62.0%",
      pub_rec: 0,
      collections_12_mths_ex_med: 0
    },
    subprime: {
      loan_amnt: 35000,
      term: " 60 months",
      int_rate: 24.99,
      annual_inc: 40000,
      emp_length: "< 1 year",
      home_ownership: "RENT",
      verification_status: "Not Verified",
      purpose: "debt_consolidation",
      fico_range_low: 620,
      fico_range_high: 624,
      dti: 42.0,
      earliest_cr_line: "May-2015",
      inq_last_6mths: 4,
      delinq_2yrs: 3,
      open_acc: 6,
      total_acc: 10,
      revol_bal: 28000,
      revol_util: "94.2%",
      pub_rec: 2,
      collections_12_mths_ex_med: 1
    }
  };

  function applyPreset(data) {
    for (const [key, val] of Object.entries(data)) {
      const el = document.getElementById(key);
      if (el) el.value = val;
    }
    updateFinancialEstimates();
  }

  document.getElementById("btnPresetPrime").addEventListener("click", () => applyPreset(PRESETS.prime));
  document.getElementById("btnPresetNearPrime").addEventListener("click", () => applyPreset(PRESETS.nearPrime));
  document.getElementById("btnPresetSubprime").addEventListener("click", () => applyPreset(PRESETS.subprime));

  // Form Submit & Scoring
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const btnText = submitBtn.querySelector(".btn-text");
    const btnSpinner = submitBtn.querySelector(".btn-spinner");

    btnText.textContent = "Evaluating Risk Profile...";
    btnSpinner.style.display = "inline";
    submitBtn.disabled = true;

    const formData = new FormData(form);
    const payload = {};
    formData.forEach((val, key) => {
      if (["loan_amnt", "int_rate", "installment", "annual_inc", "dti", "revol_bal"].includes(key)) {
        payload[key] = parseFloat(val);
      } else if (["delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "total_acc", "fico_range_low", "fico_range_high", "collections_12_mths_ex_med"].includes(key)) {
        payload[key] = parseInt(val, 10);
      } else {
        payload[key] = val;
      }
    });

    try {
      const res = await fetch(API_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`);
      }

      const jsonResponse = await res.json();
      renderDecision(jsonResponse.data);
    } catch (err) {
      console.warn("API request failed, calculating client-side fallback:", err);
      // Client-side fallback computation
      const fallbackResult = calculateClientFallback(payload);
      renderDecision(fallbackResult);
    } finally {
      btnText.textContent = "Score Loan Application";
      btnSpinner.style.display = "none";
      submitBtn.disabled = false;
    }
  });

  function renderDecision(data) {
    const pd = data.probability_of_default;
    const pdPct = (pd * 100).toFixed(2);
    
    // Animate SVG Gauge
    const arcLength = 251.2;
    const offset = arcLength * (1 - Math.min(1.0, Math.max(0.0, pd)));
    gaugeFill.style.strokeDashoffset = offset;

    pdPercentage.textContent = `${pdPct}%`;
    creditScoreText.textContent = data.internal_credit_score;
    riskTierText.textContent = data.risk_tier;
    rawJsonDisplay.textContent = JSON.stringify(data, null, 2);

    // Reset styles
    decisionBadge.className = "badge";
    recommendationBox.className = "recommendation-box";

    if (pd <= 0.10) {
      gaugeFill.style.stroke = "var(--color-green)";
      decisionBadge.classList.add("green");
      decisionBadge.textContent = "Approved";
      recommendationBox.classList.add("green");
      recIcon.textContent = "✅";
      recTitle.textContent = "Instant Approval (Prime Tier)";
      recDesc.textContent = "Applicant satisfies low-risk underwriting thresholds. Recommended for lowest interest tier.";
    } else if (pd <= 0.25) {
      gaugeFill.style.stroke = "var(--color-amber)";
      decisionBadge.classList.add("yellow");
      decisionBadge.textContent = "Manual Review";
      recommendationBox.classList.add("yellow");
      recIcon.textContent = "⚠️";
      recTitle.textContent = "Conditional / Stricter Terms";
      recDesc.textContent = "Moderate default propensity detected. Recommend manual underwriting review or collateral/co-signer.";
    } else {
      gaugeFill.style.stroke = "var(--color-red)";
      decisionBadge.classList.add("red");
      decisionBadge.textContent = "Declined";
      recommendationBox.classList.add("red");
      recIcon.textContent = "🚫";
      recTitle.textContent = "Decline Application (Subprime)";
      recDesc.textContent = "Elevated likelihood of default exceeds institutional risk tolerance.";
    }

    // Render risk factors
    riskDriversList.innerHTML = "";
    if (data.key_risk_drivers && data.key_risk_drivers.length > 0) {
      data.key_risk_drivers.forEach((driver) => {
        const li = document.createElement("li");
        li.textContent = driver;
        if (driver.includes("High") || driver.includes("Low FICO") || driver.includes("Elevated")) {
          li.classList.add(pd > 0.25 ? "danger" : "warn");
        }
        riskDriversList.appendChild(li);
      });
    } else {
      riskDriversList.innerHTML = "<li>Strong credit fundamentals across primary indicators</li>";
    }
  }

  // Pure client-side scoring fallback if API server is offline
  function calculateClientFallback(p) {
    const ficoAvg = (p.fico_range_low + p.fico_range_high) / 2;
    const score = (
      (p.int_rate - 13.0) * 0.10 +
      (p.dti - 18.0) * 0.05 +
      (700 - ficoAvg) * 0.015 +
      p.delinq_2yrs * 0.3 +
      p.inq_last_6mths * 0.2 +
      (p.loan_amnt / (p.annual_inc + 1e-5)) * 1.5
    );
    const pd = 1 / (1 + Math.exp(-score));
    const creditScore = Math.max(300, Math.min(850, Math.round(850 - (pd * 550))));

    let tier = "Subprime (High Risk)";
    let action = "Decline";
    let color = "red";
    if (pd <= 0.10) {
      tier = "Prime (Low Risk)";
      action = "Approve";
      color = "green";
    } else if (pd <= 0.25) {
      tier = "Near Prime (Moderate Risk)";
      action = "Manual Review / Stricter Terms";
      color = "yellow";
    }

    const drivers = [];
    if (ficoAvg < 660) drivers.push(`Low FICO Score (${Math.round(ficoAvg)})`);
    if (p.dti > 35.0) drivers.push(`Elevated Debt-to-Income Ratio (${p.dti.toFixed(1)}%)`);
    if (p.inq_last_6mths >= 3) drivers.push(`High Recent Credit Inquiries (${p.inq_last_6mths} in last 6 mos)`);
    if (parseFloat(p.revol_util) > 75.0) drivers.push(`High Revolving Credit Utilization (${p.revol_util})`);
    if (drivers.length === 0) drivers.push("Strong credit fundamentals across primary indicators");

    return {
      probability_of_default: parseFloat(pd.toFixed(4)),
      probability_of_default_pct: `${(pd * 100).toFixed(2)}%`,
      risk_tier: tier,
      decision: action,
      color_badge: color,
      internal_credit_score: creditScore,
      key_risk_drivers: drivers
    };
  }

  // Trigger initial calculation
  updateFinancialEstimates();
});

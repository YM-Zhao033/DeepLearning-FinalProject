/* ═══════════════════════════════════════════════════════════
   HAR Health Assistant — Dashboard JS
   ═══════════════════════════════════════════════════════════ */

const API = '/api';
let stateChart, calorieChart, trendChart, trendCalChart, activityPieChart;
let demoRunning = false;
let pollTimer = null;
let simTime = 0;
let simInterval = null;

document.addEventListener('DOMContentLoaded', () => {
  initNav();
  initButtons();
  initCharts();
  initSimulationControls();
  checkLLM();
  startPolling();
  loadTrendData();
});

function initNav() {
  document.querySelectorAll('.nav-item').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const section = link.dataset.section;
      if (!section) return;
      document.querySelectorAll('.nav-item').forEach(l => l.classList.remove('active'));
      link.classList.add('active');
      document.querySelectorAll('.content-section').forEach(s => s.classList.remove('active'));
      const target = document.getElementById('section-' + section);
      if (target) target.classList.add('active');
      const titles = { overview: '概览', activity: '活动监测', goals: '目标追踪', trend: '趋势分析', llm: 'AI 健康助手' };
      document.getElementById('pageTitle').textContent = titles[section] || section;
      if (section === 'trend') loadTrendData();
    });
  });
}

function initButtons() {
  document.getElementById('demoBtn').addEventListener('click', toggleDemo);
  document.getElementById('resetBtn').addEventListener('click', resetSession);
  document.getElementById('checkLlmBtn').addEventListener('click', checkLLM);
  document.getElementById('llmSendBtn').addEventListener('click', sendLlmMessage);
  document.getElementById('llmChatInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendLlmMessage();
  });
}

function initSimulationControls() {
  document.querySelectorAll('.sim-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.sim-btn').forEach(b => b.classList.remove('active-sim'));
      btn.classList.add('active-sim');
    });
  });
  const durationSlider = document.getElementById('simDuration');
  const durationVal = document.getElementById('simDurationVal');
  durationSlider.addEventListener('input', () => {
    durationVal.textContent = durationSlider.value + 's';
  });
  document.getElementById('injectBtn').addEventListener('click', injectActivity);
}

async function injectActivity() {
  const activity = document.querySelector('.sim-btn.active-sim').dataset.activity;
  const durationS = parseInt(document.getElementById('simDuration').value);
  for (let i = 0; i < durationS; i++) {
    await fetch(`${API}/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ activity, duration_s: 1, sim_s: simTime }),
    });
    simTime += 2.56;
    updateSimTime();
    await delay(10);
  }
  await refreshStatus();
}

async function toggleDemo() {
  const btn = document.getElementById('demoBtn');
  const btnText = document.getElementById('demoBtnText');
  if (!demoRunning) {
    const res = await fetch(`${API}/demo/start`, { method: 'POST' });
    const data = await res.json();
    if (data.ok) {
      demoRunning = true;
      btn.classList.add('running');
      btnText.textContent = '停止演示';
      startSimTime();
    }
  } else {
    await fetch(`${API}/demo/stop`, { method: 'POST' });
    demoRunning = false;
    btn.classList.remove('running');
    btnText.textContent = '开始演示';
    stopSimTime();
  }
}

async function resetSession() {
  await fetch(`${API}/reset`, { method: 'POST' });
  simTime = 0;
  updateSimTime();
  await refreshStatus();
  stopSimTime();
  demoRunning = false;
  const btn = document.getElementById('demoBtn');
  const btnText = document.getElementById('demoBtnText');
  btn.classList.remove('running');
  btnText.textContent = '开始演示';
}

function startPolling() {
  refreshStatus();
  pollTimer = setInterval(refreshStatus, 1500);
}

async function refreshStatus() {
  try {
    const res = await fetch(`${API}/status`);
    const data = await res.json();
    updateUI(data);
    updateStateMachine(data.current_state);
  } catch (e) {
    console.warn('Status poll failed:', e);
  }
}

function updateUI(data) {
  const state = data.current_state || 'IDLE';
  const activity = data.current_activity || state;

  document.getElementById('stateName').textContent = state;
  document.getElementById('stateActivity').textContent = activity;

  const badgeText = document.getElementById('stateBadgeText');
  const badge = document.getElementById('stateBadge');
  const ring = document.querySelector('.state-ring');
  const iconArea = document.getElementById('stateIconArea');

  badge.className = 'state-badge';
  ring.className = 'state-ring';
  iconArea.className = 'state-icon';

  if (state === 'ACTIVE') {
    badge.classList.add('active-badge');
    badgeText.textContent = '运动中';
    ring.classList.add('active-ring');
    iconArea.classList.add('active-icon');
  } else if (state === 'IDLE') {
    badgeText.textContent = '低功耗';
    ring.classList.add('idle-ring');
    iconArea.classList.add('idle-icon');
  } else {
    badgeText.textContent = '检测中';
    ring.classList.add('check-ring');
    iconArea.classList.add('check-icon');
  }

  const cal = data.calories_kcal || 0;
  const active = data.active_min || 0;
  const sitting = data.sitting_min || 0;
  const gp = data.goal_progress || {};
  const intensity = data.intensity || {};

  document.getElementById('calValue').textContent = cal.toFixed(1);
  document.getElementById('activeValue').textContent = active.toFixed(1);
  document.getElementById('sittingValue').textContent = sitting.toFixed(1);

  const calPct = Math.min(gp.cal_pct || 0, 100);
  const activePct = Math.min(gp.active_pct || 0, 100);
  document.getElementById('calProgress').style.width = calPct + '%';
  document.getElementById('activeProgress').style.width = activePct + '%';
  document.getElementById('calGoal').textContent = `目标: ${gp.cal_goal || 200} kcal`;
  document.getElementById('activeGoal').textContent = `目标: ${gp.active_goal || 30} min`;

  const sitAlert = document.getElementById('sittingAlert');
  const sitPct = Math.min((sitting / 45) * 100, 100);
  document.getElementById('sittingProgress').style.width = sitPct + '%';
  if (sitting > 45) {
    sitAlert.textContent = '⚠️ 久坐警告';
    sitAlert.style.color = 'var(--danger)';
  } else if (sitting > 20) {
    sitAlert.textContent = '💡 起身活动';
    sitAlert.style.color = 'var(--warning)';
  } else {
    sitAlert.textContent = '无久坐警告';
    sitAlert.style.color = '';
  }

  const gradeLabel = intensity.grade_label || '—';
  document.getElementById('intensityValue').textContent = gradeLabel;
  document.getElementById('intensityDetail').textContent =
    `中等 ${intensity.moderate_min || 0} min | 高强度 ${intensity.vigorous_min || 0} min`;

  const alerts = data.summary?.alerts || [];
  const alertsContainer = document.getElementById('alertsContainer');
  const placeholder = document.getElementById('alertPlaceholder');
  if (alerts.length === 0) {
    placeholder.style.display = 'flex';
  } else {
    placeholder.style.display = 'none';
    alerts.forEach(alertText => {
      const existing = [...alertsContainer.querySelectorAll('.alert-item')]
        .find(el => el.textContent.includes(alertText.slice(0, 10)));
      if (!existing) {
        const alertEl = document.createElement('div');
        let cls = 'alert-item info-alert';
        if (alertText.includes('警告') || alertText.includes('⚠')) cls = 'alert-item warning-alert';
        else if (alertText.includes('运动') || alertText.includes('继续保持')) cls = 'alert-item success-alert';
        alertEl.className = cls;
        alertEl.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4m0 4h.01"/></svg><span>${alertText}</span>`;
        alertsContainer.insertBefore(alertEl, placeholder);
        setTimeout(() => alertEl.remove(), 10000);
      }
    });
  }

  const tip = data.summary?.alerts?.[0] || '';
  if (tip) document.getElementById('llmTipBody').textContent = tip;

  const actBadge = document.getElementById('activityBadge');
  actBadge.textContent = activity;
  actBadge.className = 'activity-badge';
  if (state === 'ACTIVE') actBadge.classList.add('active-badge');
  else if (state === 'IDLE') actBadge.classList.add('idle-badge');
  else actBadge.classList.add('check-badge');

  updateStateChart(data.summary?.state_counts || {});
  updateCalorieChart(data);
  updateGoals(data);
  updateWHO(data);
}

function updateStateMachine(state) {
  document.querySelectorAll('.sm-circle').forEach(c => c.classList.remove('active'));
  document.querySelectorAll('.sm-arrow').forEach(a => a.classList.remove('active'));
  if (state === 'ACTIVE') {
    document.querySelector('.active-circle')?.classList.add('active');
    document.getElementById('arrowIdleCheck')?.classList.add('active');
    document.getElementById('arrowCheckActive')?.classList.add('active');
  } else if (state === 'CHECK') {
    document.querySelector('.check-circle')?.classList.add('active');
    document.getElementById('arrowIdleCheck')?.classList.add('active');
  } else if (state === 'IDLE') {
    document.querySelector('.idle-circle')?.classList.add('active');
  }
}

function initCharts() {
  const defaultConfig = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    animation: { duration: 400 },
  };

  stateChart = new Chart(document.getElementById('stateChart'), {
    type: 'doughnut',
    data: {
      labels: ['IDLE', 'CHECK', 'ACTIVE'],
      datasets: [{ data: [0, 0, 0], backgroundColor: ['#f59e0b', '#3b82f6', '#22c55e'], borderWidth: 0, hoverOffset: 6 }],
    },
    options: { ...defaultConfig, cutout: '65%', plugins: { legend: { display: true, position: 'bottom', labels: { font: { size: 12 }, padding: 16 } } } },
  });

  calorieChart = new Chart(document.getElementById('calorieChart'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [{ label: '累计卡路里 (kcal)', data: [], borderColor: '#4f46e5', backgroundColor: 'rgba(79,70,229,0.1)', fill: true, tension: 0.4, pointRadius: 3 }],
    },
    options: { ...defaultConfig, scales: { x: { grid: { display: false }, ticks: { font: { size: 11 }, color: '#9097a5' } }, y: { beginAtZero: true, grid: { color: '#e4e7ed' }, ticks: { font: { size: 11 }, color: '#9097a5' } } } },
  });

  trendChart = new Chart(document.getElementById('trendChart'), {
    type: 'bar',
    data: {
      labels: [],
      datasets: [{ label: '活动分钟数', data: [], backgroundColor: '#22c55e', borderRadius: 6, barPercentage: 0.6 }],
    },
    options: { ...defaultConfig, scales: { x: { grid: { display: false }, ticks: { font: { size: 11 }, color: '#9097a5' } }, y: { beginAtZero: true, grid: { color: '#e4e7ed' }, ticks: { font: { size: 11 }, color: '#9097a5' } } } },
  });

  trendCalChart = new Chart(document.getElementById('trendCalChart'), {
    type: 'bar',
    data: {
      labels: [],
      datasets: [{ label: '卡路里消耗', data: [], backgroundColor: '#4f46e5', borderRadius: 6, barPercentage: 0.6 }],
    },
    options: { ...defaultConfig, scales: { x: { grid: { display: false }, ticks: { font: { size: 11 }, color: '#9097a5' } }, y: { beginAtZero: true, grid: { color: '#e4e7ed' }, ticks: { font: { size: 11 }, color: '#9097a5' } } } },
  });

  activityPieChart = new Chart(document.getElementById('activityPieChart'), {
    type: 'pie',
    data: {
      labels: ['WALKING', 'W_UP', 'W_DOWN', 'SITTING', 'STANDING', 'LAYING'],
      datasets: [{ data: [0, 0, 0, 0, 0, 0], backgroundColor: ['#22c55e', '#10b981', '#06b6d4', '#f59e0b', '#eab308', '#8b5cf6'], borderWidth: 0, hoverOffset: 6 }],
    },
    options: { ...defaultConfig, plugins: { legend: { display: true, position: 'bottom', labels: { font: { size: 12 }, padding: 12 } } } },
  });
}

function updateStateChart(counts) {
  if (!stateChart) return;
  stateChart.data.datasets[0].data = [counts.IDLE || 0, counts.CHECK || 0, counts.ACTIVE || 0];
  stateChart.update('none');
}

function updateCalorieChart(data) {
  if (!calorieChart) return;
  const hist = data.summary?._history || [];
  calorieChart.data.labels = hist.map((_, i) => i);
  calorieChart.data.datasets[0].data = hist.map(h => h.cumulative);
  calorieChart.update('none');
}

async function loadTrendData() {
  try {
    const res = await fetch(`${API}/activity_log`);
    const days = await res.json();
    const labels = days.map(d => d.date ? d.date.slice(5) : '');
    const activeVals = days.map(d => d.total_active_min || 0);
    const calVals = days.map(d => d.total_cal || 0);

    if (trendChart) {
      trendChart.data.labels = labels;
      trendChart.data.datasets[0].data = activeVals;
      trendChart.update();
    }
    if (trendCalChart) {
      trendCalChart.data.labels = labels;
      trendCalChart.data.datasets[0].data = calVals;
      trendCalChart.update();
    }

    const activityCounts = { WALKING: 0, WALKING_UPSTAIRS: 0, WALKING_DOWNSTAIRS: 0, SITTING: 0, STANDING: 0, LAYING: 0 };
    days.forEach(d => {
      const acts = d.activities || {};
      Object.keys(acts).forEach(k => { if (k in activityCounts) activityCounts[k] += acts[k]; });
    });

    if (activityPieChart) {
      activityPieChart.data.datasets[0].data = [
        activityCounts.WALKING || 0, activityCounts.WALKING_UPSTAIRS || 0,
        activityCounts.WALKING_DOWNSTAIRS || 0, activityCounts.SITTING || 0,
        activityCounts.STANDING || 0, activityCounts.LAYING || 0,
      ];
      activityPieChart.update();
    }

    const trendRes = await fetch(`${API}/trend`);
    const trend = await trendRes.json();
    updateTrendSummary(trend);
  } catch (e) {
    console.warn('Trend load failed:', e);
  }
}

function updateTrendSummary(trend) {
  const dirEl = document.getElementById('trendDir');
  const changeEl = document.getElementById('trendChange');
  const arrowEl = document.getElementById('trendArrow');
  const dir = trend.trend_direction || 'stable';
  const change = trend.change_percent || 0;
  const labels = { up: '上升', down: '下降', stable: '稳定' };
  dirEl.textContent = labels[dir] || dir;
  changeEl.textContent = (change >= 0 ? '+' : '') + change + '%';
  arrowEl.className = 'trend-arrow';
  if (dir === 'up') arrowEl.classList.add('up');
  else if (dir === 'down') arrowEl.classList.add('down');
}

function updateGoals(data) {
  const gp = data.goal_progress || {};
  const activePct = gp.active_pct || 0;
  const calPct = gp.cal_pct || 0;
  document.getElementById('activeGoalPct').textContent = Math.round(activePct) + '%';
  document.getElementById('calGoalPct').textContent = Math.round(calPct) + '%';
  document.getElementById('goalActiveAchieved').textContent = (gp.achieved_active || 0).toFixed(1);
  document.getElementById('goalActiveTarget').textContent = gp.active_goal || 30;
  document.getElementById('goalActiveRemaining').textContent =
    (gp.active_remaining || 0) > 0 ? `还差 ${(gp.active_remaining || 0).toFixed(1)} 分钟` : '已达成!';
  document.getElementById('goalCalAchieved').textContent = (gp.achieved_cal || 0).toFixed(1);
  document.getElementById('goalCalTarget').textContent = gp.cal_goal || 200;
  document.getElementById('goalCalRemaining').textContent =
    (gp.cal_remaining || 0) > 0 ? `还差 ${(gp.cal_remaining || 0).toFixed(1)} kcal` : '已达成!';
  const circ = 327;
  document.getElementById('activeRing').style.strokeDasharray = (activePct / 100 * circ) + ' ' + circ;
  document.getElementById('calRing').style.strokeDasharray = (calPct / 100 * circ) + ' ' + circ;
}

function updateWHO(data) {
  const grade = data.intensity?.grade || 'insufficient';
  const levels = ['active', 'moderate', 'insufficient'];
  levels.forEach(l => {
    const el = document.getElementById('who' + l.charAt(0).toUpperCase() + l.slice(1));
    if (el) { el.className = 'who-level'; }
    const check = document.getElementById('who' + l.charAt(0).toUpperCase() + l.slice(1) + 'Check');
    if (check) check.textContent = '未达成';
  });
  const activeEl = document.getElementById('whoActive');
  const activeCheck = document.getElementById('whoActiveCheck');
  const modEl = document.getElementById('whoModerate');
  const modCheck = document.getElementById('whoModerateCheck');
  const insuffEl = document.getElementById('whoInsufficient');
  if (grade === 'active') {
    activeEl?.classList.add('active-level');
    if (activeCheck) activeCheck.textContent = '已达成 ✓';
  } else if (grade === 'moderate') {
    modEl?.classList.add('active-level');
    if (modCheck) modCheck.textContent = '已达成 ✓';
  } else {
    insuffEl?.classList.add('active-level');
  }
}

async function checkLLM() {
  const statusEl = document.getElementById('llmConnStatus');
  const dotEl = document.querySelector('.llm-dot');
  const modelEl = document.getElementById('llmModelName');
  statusEl.textContent = '检测中...';
  try {
    const res = await fetch(`${API}/llm_available`);
    const data = await res.json();
    if (data.available) {
      statusEl.textContent = '已连接';
      dotEl.classList.add('online');
      dotEl.classList.remove('offline');
      modelEl.textContent = data.models?.[0] || 'qwen2.5:7b';
    } else {
      statusEl.textContent = '未连接 (Ollama离线)';
      dotEl.classList.add('offline');
      dotEl.classList.remove('online');
    }
  } catch (e) {
    statusEl.textContent = '连接失败';
    dotEl.classList.add('offline');
    dotEl.classList.remove('online');
  }
}

function sendLlmMessage() {
  const input = document.getElementById('llmChatInput');
  const body = document.getElementById('llmChatBody');
  const text = input.value.trim();
  if (!text) return;
  const userMsg = document.createElement('div');
  userMsg.className = 'chat-message';
  userMsg.style.cssText = 'flex-direction:row-reverse;';
  userMsg.innerHTML = `<div class="chat-bubble" style="background:var(--primary);color:#fff;border:none;max-width:75%;">${escapeHtml(text)}</div>`;
  body.appendChild(userMsg);
  input.value = '';
  const botMsg = document.createElement('div');
  botMsg.className = 'chat-message bot';
  botMsg.innerHTML = `<div class="chat-avatar bot-avatar"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M12 1v4m0 14v4M4.22 4.22l2.83 2.83m9.9 9.9 2.83 2.83M1 12h4m14 0h4M4.22 19.78l2.83-2.83m9.9-9.9 2.83-2.83"/></svg></div><div class="chat-bubble bot-bubble">分析中...</div>`;
  body.appendChild(botMsg);
  body.scrollTop = body.scrollHeight;
  document.getElementById('llmTipBody').textContent = 'AI 正在分析您的请求...';
  fetch(`${API}/llm_tip`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ language: 'zh', query: text }),
  }).then(r => r.json()).then(data => {
    const tip = data.tip || '抱歉，AI 助手暂时无法响应。';
    botMsg.querySelector('.bot-bubble').textContent = tip;
    document.getElementById('llmTipBody').textContent = tip;
    body.scrollTop = body.scrollHeight;
  }).catch(() => {
    botMsg.querySelector('.bot-bubble').textContent = '连接失败，请检查 Ollama 是否运行中。';
  });
}

function startSimTime() {
  if (simInterval) clearInterval(simInterval);
  simInterval = setInterval(() => { simTime += 2.56; updateSimTime(); }, 500);
}

function stopSimTime() {
  if (simInterval) { clearInterval(simInterval); simInterval = null; }
}

function updateSimTime() {
  const h = Math.floor(simTime / 3600);
  const m = Math.floor((simTime % 3600) / 60);
  const s = Math.floor(simTime % 60);
  document.getElementById('simTimeValue').textContent =
    `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
}

function delay(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

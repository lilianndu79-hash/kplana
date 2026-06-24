// ==================== KPL 分析面板 - 前端逻辑 ====================

// 全局状态
let heroes = [], teams = [], currentLeague = '20260003';
let radarChart = null, heroTrendChart = null;

// ==================== 初始化 ====================

async function api(url, opts) {
  const r = await fetch(url, opts);
  return r.json();
}

async function init() {
  // 赛季列表
  const leagues = await api('/api/leagues');
  const sel = document.getElementById('leagueSelect');
  leagues.forEach(l => {
    const o = document.createElement('option');
    o.value = l.league_id;
    o.textContent = l.league_name;
    if (l.league_id === currentLeague) o.selected = true;
    sel.appendChild(o);
  });

  // 英雄列表
  heroes = await api('/api/heroes');
  function fillHeroSelect(el) {
    if (!el) return;
    heroes.forEach(h => {
      const o = document.createElement('option');
      o.value = h.hero_id;
      o.textContent = h.hero_name;
      el.appendChild(o);
    });
  }
  document.querySelectorAll('.bp-hero-a, .bp-hero-b').forEach(fillHeroSelect);
  fillHeroSelect(document.getElementById('heroSelect'));

  // 战队列表
  teams = await api('/api/teams');
  ['teamASelect', 'teamBSelect'].forEach(id => {
    const s = document.getElementById(id);
    if (!s) return;
    teams.forEach(t => {
      const o = document.createElement('option');
      o.value = t.team_name;
      o.textContent = t.team_name;
      s.appendChild(o);
    });
  });

  const teamSel = document.getElementById('teamSelect');
  if (teamSel) {
    teams.forEach(t => {
      const o = document.createElement('option');
      o.value = t.team_id;
      o.textContent = t.team_name;
      teamSel.appendChild(o);
    });
  }

  // 选手搜索列表(datalist)
  const players = await api('/api/players?league_id=' + currentLeague);
  const dl = document.getElementById('playerList');
  if (dl) {
    players.forEach(p => {
      const o = document.createElement('option');
      o.value = p.player_name;
      dl.appendChild(o);
    });
  }

  loadOverview();
}

// ==================== 标签切换 ====================

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.nav button').forEach(b => b.classList.remove('active'));
  const tab = document.getElementById('tab-' + name);
  if (tab) tab.classList.add('active');
  const btn = document.querySelector(`.nav button[onclick*="${name}"]`);
  if (btn) btn.classList.add('active');

  if (name === 'overview') loadOverview();
  if (name === 'team') loadTeamSelect();
  if (name === 'hero') loadHeroSelect();
}

function switchLeague() {
  currentLeague = document.getElementById('leagueSelect').value;
  loadOverview();
}

// ==================== 赛季总览 ====================

async function loadOverview() {
  const div = document.getElementById('overviewContent');
  div.innerHTML = '<div class="loading">加载中...</div>';

  const data = await api('/api/overview?league_id=' + currentLeague);

  // 战队排行
  let ranking = '<div class="card"><h3>战队排行</h3><table><tr><th>#</th><th>战队</th><th>大场胜率</th><th>小局数</th><th>胜/负</th><th>场均KDA</th><th>分均经济</th><th>一血率</th><th>暴君控制率</th></tr>';
  data.ranking.forEach((r, i) => {
    ranking += `<tr><td>${i+1}</td><td>${r.team_name}</td><td class="${(r.win_rate||0)>0.5?'green':'red'}">${((r.win_rate||0)*100).toFixed(1)}%</td>`
      + `<td>${r.battle_count||0}</td><td>${r.wins||0}/${r.losses||0}</td>`
      + `<td>${r.avg_kda||'--'}</td><td>${r.avg_gpm||'--'}</td>`
      + `<td>${((r.avg_first_blood_cnt||0)*100).toFixed(0)}%</td>`
      + `<td>${((r.avg_tyrant_control_rate||0)*100).toFixed(0)}%</td></tr>`;
  });
  ranking += '</table></div>';

  // 英雄热度 (用柱状图)
  let heroesHtml = '<div class="card"><h3>英雄热度 TOP10</h3><div class="row">';
  data.hot_heroes.slice(0, 10).forEach(h => {
    const wr = ((h.win_rate || 0) * 100).toFixed(0);
    const maxPick = Math.max(...data.hot_heroes.map(x => x.pick_count || 0), 1);
    heroesHtml += `<div style="flex:0 0 100%;margin:2px 0"><div class="bar-row">`
      + `<span style="width:30px;font-size:12px">${h.hero_name.slice(0,1)}</span>`
      + `<span style="width:90px;font-size:12px">${h.hero_name}</span>`
      + `<div style="flex:1;background:#1a1f3e;border-radius:3px">`
      + `<div class="bar-fill bar-gold" style="width:${(h.pick_count/maxPick*100).toFixed(0)}%"></div></div>`
      + `<span style="width:50px;font-size:12px">P:${h.pick_count}</span>`
      + `<span style="width:50px;font-size:12px">B:${h.ban_count}</span>`
      + `<span style="width:45px;font-size:12px;color:${wr>50?'#4caf50':'#f44336'}">${wr}%</span></div></div>`;
  });
  heroesHtml += '</div></div>';

  div.innerHTML = ranking + heroesHtml;
}

// ==================== BP 模拟器 ====================

async function updateBP() {
  const hA = Array.from(document.querySelectorAll('.bp-hero-a')).map(s => parseInt(s.value)).filter(v => v);
  const hB = Array.from(document.querySelectorAll('.bp-hero-b')).map(s => parseInt(s.value)).filter(v => v);
  const tA = document.getElementById('teamASelect').value;
  const tB = document.getElementById('teamBSelect').value;

  if (hA.length !== 5 || hB.length !== 5) {
    document.getElementById('bpResult').innerHTML = `<h3>预测结果</h3><div class="loading">蓝方已选${hA.length}/5 红方已选${hB.length}/5</div>`;
    document.getElementById('bpDetail').innerHTML = '';
    return;
  }

  const resp = await fetch('/api/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({team_a: {name: tA, heroes: hA}, team_b: {name: tB, heroes: hB}})
  });
  const data = await resp.json();

  const wr = data.prediction.win_rate_a;
  const conf = data.prediction.confidence;
  const wrColor = wr > 0.55 ? '#4caf50' : wr < 0.45 ? '#f44336' : '#e6b422';

  document.getElementById('bpResult').innerHTML = `
    <h3>预测结果 <span class="small">(${data.prediction.model})</span></h3>
    <div class="result">
      <div style="color:${wrColor};font-size:48px">${(wr*100).toFixed(1)}%</div>
      <div style="font-size:14px;color:#888">蓝方胜率 | 置信度: ${(conf*100).toFixed(0)}%</div>
    </div>
    <div style="font-size:12px;color:#888;margin-top:8px">
      ${Object.entries(data.breakdown).map(([k,v]) => `<div>${k}: ${(v*100).toFixed(0)}%</div>`).join('')}
    </div>`;

  // 协同 + 克制
  let detail = '';
  ['team_a', 'team_b'].forEach((side, si) => {
    const syns = (data.synergy[side] || []).filter(s => s.win_rate !== null);
    const avg = syns.length ? (syns.reduce((a,b) => a + b.win_rate, 0) / syns.length * 100).toFixed(0) : '--';
    detail += `<div class="col"><div class="card"><h3>${si===0?'蓝方':'红方'} 阵容协同 (均${avg}%)</h3>`;
    syns.forEach(s => {
      const ha = data.heroes_a.find(x => x.id === s.hero_a) || data.heroes_b.find(x => x.id === s.hero_a);
      const hb = data.heroes_a.find(x => x.id === s.hero_b) || data.heroes_b.find(x => x.id === s.hero_b);
      const wr = ((s.win_rate || 0) * 100).toFixed(0);
      detail += `<div class="bar-row">
        <span style="width:80px">${ha?ha.name:s.hero_a}</span><span>+</span>
        <span style="width:80px">${hb?hb.name:s.hero_b}</span>
        <div style="flex:1;background:#1a1f3e;border-radius:3px">
          <div class="bar-fill ${wr>55?'bar-green':'bar-red'}" style="width:${Math.min(wr,100)}%"></div></div>
        <span style="width:60px">${wr}%</span>
        <span class="small">${s.games}场</span></div>`;
    });
    detail += '</div></div>';
  });
  document.getElementById('bpDetail').innerHTML = detail;
}

// ==================== 选手分析 ====================

let playerTimer;
function searchPlayer() {
  clearTimeout(playerTimer);
  playerTimer = setTimeout(async () => {
    const name = document.getElementById('playerSearch').value.trim();
    if (!name) { document.getElementById('playerContent').innerHTML = '<div class="loading">请输入选手名</div>'; return; }
    document.getElementById('playerContent').innerHTML = '<div class="loading">搜索中...</div>';

    try {
      const data = await api('/api/player/' + encodeURIComponent(name));
      if (data.error) { document.getElementById('playerContent').innerHTML = `<div class="loading">${data.error}</div>`; return; }
      renderPlayer(data);
    } catch(e) {
      document.getElementById('playerContent').innerHTML = `<div class="loading">搜索出错: ${e}</div>`;
    }
  }, 300);
}

function renderPlayer(data) {
  const p = data.info;
  let html = `<div class="card">
    <div class="row">
      <div style="text-align:center;min-width:80px">
        <img src="${p.player_icon}" width="60" onerror="this.style.display='none'" style="border-radius:50%">
        <div style="font-weight:bold;margin-top:4px">${p.full_name || p.player_name}</div>
      </div>
      <div style="flex:1">`;

  if (data.position_dist.length) {
    data.position_dist.forEach(d => { html += `<span class="hero-tag">${d.position_desc}: ${d.cnt}场</span>`; });
  }
  html += `</div></div></div>`;

  // 五维雷达图
  if (data.season_stats.length) {
    const latest = data.season_stats[0];
    html += `<div class="row">
      <div class="col"><div class="card"><h3>能力雷达</h3><div class="chart-box"><canvas id="radarChart"></canvas></div></div></div>
      <div class="col"><div class="card"><h3>赛季数据</h3>
        <div class="stat-grid">
          <div class="stat-item"><div class="val green">${latest.games||0}</div><div class="lbl">出场</div></div>
          <div class="stat-item"><div class="val green">${latest.wins||0}</div><div class="lbl">胜场</div></div>
          <div class="stat-item"><div class="val gold">${latest.avg_kda||0}</div><div class="lbl">KDA</div></div>
          <div class="stat-item"><div class="val gold">${latest.avg_mvp||0}</div><div class="lbl">MVP分</div></div>
          <div class="stat-item"><div class="val">${((latest.avg_part||0)*100).toFixed(0)}%</div><div class="lbl">参团率</div></div>
          <div class="stat-item"><div class="val">${latest.avg_gold||0}</div><div class="lbl">场均经济</div></div>
        </div>
      </div></div></div>`;
  }

  // 赛季统计表
  if (data.season_stats.length > 1) {
    html += `<div class="card"><h3>赛季表现</h3><table><tr><th>赛季</th><th>场次</th><th>胜场</th><th>胜率</th><th>KDA</th><th>MVP</th><th>参团率</th><th>经济</th><th>输出</th></tr>`;
    data.season_stats.forEach(s => {
      html += `<tr><td>${s.league_name}</td><td>${s.games}</td><td>${s.wins}</td>
        <td class="${(s.wins/s.games)>0.5?'green':'red'}">${(s.wins/s.games*100).toFixed(0)}%</td>
        <td>${s.avg_kda}</td><td>${s.avg_mvp}</td><td>${((s.avg_part||0)*100).toFixed(0)}%</td>
        <td>${s.avg_gold||'--'}</td><td>${s.avg_damage||'--'}</td></tr>`;
    });
    html += '</table></div>';
  }

  // 英雄池
  if (data.hero_pool.length) {
    html += `<div class="card"><h3>英雄池</h3><div class="row">`;
    data.hero_pool.slice(0, 15).forEach(h => {
      const wr = ((h.win_rate||0)*100).toFixed(0);
      html += `<div style="flex:0 0 200px;margin:4px"><div class="bar-row">
        <span style="width:80px">${h.hero_name}</span>
        <span style="width:40px">${h.games_played}场</span>
        <div style="flex:1;background:#1a1f3e;border-radius:3px">
          <div class="bar-fill ${wr>55?'bar-green':'bar-red'}" style="width:${Math.min(wr,100)}%"></div></div>
        <span class="${wr>55?'green':'red'}" style="width:40px">${wr}%</span></div></div>`;
    });
    html += '</div></div>';
  }

  // 近期比赛
  if (data.recent_games.length) {
    html += `<div class="card"><h3>近期比赛</h3><table><tr><th>时间</th><th>英雄</th><th>位置</th>
      <th>K/D/A</th><th>KDA</th><th>MVP分</th><th>结果</th></tr>`;
    data.recent_games.forEach(g => {
      html += `<tr><td>${(g.start_time||'').substr(0,10)}</td><td>${g.hero_name}</td>
        <td>${g.position_desc}</td><td>${g.kills}/${g.deaths}/${g.assists}</td>
        <td>${g.kda}</td><td>${g.mvp_score}${g.is_mvp?' MVP':''}</td>
        <td class="${g.is_win?'green':'red'}">${g.is_win?'胜':'负'}</td></tr>`;
    });
    html += '</table></div>';
  }

  document.getElementById('playerContent').innerHTML = html;

  // 绘制雷达图
  setTimeout(drawRadar, 100);
}

function drawRadar() {
  const canvas = document.getElementById('radarChart');
  if (!canvas) return;
  if (radarChart) radarChart.destroy();

  // 从DOM提取数据
  const stats = document.querySelectorAll('.stat-item .val');
  if (stats.length < 4) return;
  const vals = Array.from(stats).slice(0, 5).map(s => {
    const v = parseFloat(s.textContent);
    if (s.textContent.includes('%')) return v;
    return isNaN(v) ? 50 : v;
  });

  const labels = ['出场', '胜场', 'KDA', 'MVP分', '参团率'];
  // 归一化
  const maxes = [30, 30, 10, 10, 100];
  const normalized = vals.map((v, i) => Math.min(v / maxes[i] * 100, 100));

  radarChart = new Chart(canvas, {
    type: 'radar',
    data: {
      labels: labels,
      datasets: [{
        label: '能力维度',
        data: normalized,
        backgroundColor: 'rgba(230,180,34,0.2)',
        borderColor: '#e6b422',
        borderWidth: 2,
        pointBackgroundColor: '#e6b422',
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          beginAtZero: true,
          max: 100,
          ticks: { display: false },
          grid: { color: 'rgba(255,255,255,0.1)' },
          pointLabels: { color: '#aaa', font: { size: 12 } }
        }
      },
      plugins: { legend: { display: false } }
    }
  });
}

// ==================== 选手对比 (新增) ====================

let compareTimer;
function searchCompare() {
  clearTimeout(compareTimer);
  compareTimer = setTimeout(async () => {
    const p1 = document.getElementById('compareP1').value.trim();
    const p2 = document.getElementById('compareP2').value.trim();
    if (!p1 || !p2) { document.getElementById('compareContent').innerHTML = '<div class="loading">请输入两个选手名</div>'; return; }

    document.getElementById('compareContent').innerHTML = '<div class="loading">对比中...</div>';
    const data = await api('/api/compare?p1=' + encodeURIComponent(p1) + '&p2=' + encodeURIComponent(p2));

    let html = '<div class="compare-cards">';
    ['player_a', 'player_b'].forEach(key => {
      const d = data[key];
      if (!d || d.error) {
        html += `<div class="card"><h3>${d?d.error:'加载失败'}</h3></div>`;
      } else {
        const s = d.stats || {};
        const wr = s.total_games ? (s.wins / s.total_games * 100).toFixed(0) : '--';
        html += `<div class="card">
          <h3>${d.info.full_name || d.info.player_name}</h3>
          <div class="stat-grid">
            <div class="stat-item"><div class="val green">${s.total_games||0}</div><div class="lbl">总场次</div></div>
            <div class="stat-item"><div class="val green">${wr}%</div><div class="lbl">胜率</div></div>
            <div class="stat-item"><div class="val gold">${s.avg_kda||0}</div><div class="lbl">KDA</div></div>
            <div class="stat-item"><div class="val gold">${s.avg_mvp||0}</div><div class="lbl">MVP分</div></div>
            <div class="stat-item"><div class="val">${((s.avg_part||0)*100).toFixed(0)}%</div><div class="lbl">参团率</div></div>
            <div class="stat-item"><div class="val">${s.avg_gold||0}</div><div class="lbl">场均经济</div></div>
            <div class="stat-item"><div class="val">${s.avg_damage||0}</div><div class="lbl">场均输出</div></div>
            <div class="stat-item"><div class="val">${s.avg_tank||0}</div><div class="lbl">场均承伤</div></div>
          </div>
        </div>`;
      }
      if (key === 'player_a') html += '<div style="text-align:center;align-self:center;font-size:32px;color:#e6b422">VS</div>';
    });
    html += '</div>';
    document.getElementById('compareContent').innerHTML = html;
  }, 400);
}

// ==================== 战队分析 ====================

async function loadTeamSelect() {
  if (document.getElementById('teamSelect').options.length > 1) return;
}

async function loadTeam() {
  const tid = document.getElementById('teamSelect').value;
  if (!tid) return;
  document.getElementById('teamContent').innerHTML = '<div class="loading">加载中...</div>';

  const data = await api('/api/team/' + tid + '?league_id=' + currentLeague);
  const t = data.info, s = data.stats || {};
  let html = `<div class="card"><h3>${t.team_name} (${t.team_abbr})</h3>`;
  if (s.win_rate !== undefined) {
    html += `<div class="stat-grid">
      <div class="stat-item"><div class="val green">${((s.win_rate||0)*100).toFixed(0)}%</div><div class="lbl">大场胜率</div></div>
      <div class="stat-item"><div class="val">${s.battle_count||0}</div><div class="lbl">小局数</div></div>
      <div class="stat-item"><div class="val gold">${s.avg_kda||'--'}</div><div class="lbl">场均KDA</div></div>
      <div class="stat-item"><div class="val">${((s.avg_first_blood_cnt||0)*100).toFixed(0)}%</div><div class="lbl">一血率</div></div>
      <div class="stat-item"><div class="val">${((s.avg_tyrant_control_rate||0)*100).toFixed(0)}%</div><div class="lbl">暴君控制率</div></div>
      <div class="stat-item"><div class="val">${((s.avg_dragon_control_rate||0)*100).toFixed(0)}%</div><div class="lbl">主宰控制率</div></div>
    </div>`;
  }
  html += '</div>';

  // 队员 + 常用英雄
  html += '<div class="row">';
  if (data.players.length) {
    html += `<div class="col"><div class="card"><h3>队员</h3><table>`;
    data.players.forEach(p => {
      html += `<tr><td>${p.player_name}</td><td>${p.position_desc}</td><td>${p.games}场</td></tr>`;
    });
    html += '</table></div></div>';
  }
  if (data.top_heroes && data.top_heroes.length) {
    html += `<div class="col"><div class="card"><h3>常用英雄</h3>`;
    data.top_heroes.forEach(h => {
      const wr = ((h.win_rate||0)*100).toFixed(0);
      html += `<div class="bar-row">
        <span style="width:80px">${h.hero_name}</span><span style="width:50px">${h.pick_count}场</span>
        <div style="flex:1;background:#1a1f3e;border-radius:3px">
          <div class="bar-fill ${wr>55?'bar-green':'bar-red'}" style="width:${Math.min(wr,100)}%"></div></div>
        <span style="width:45px">${wr}%</span></div>`;
    });
    html += '</div></div>';
  }
  html += '</div>';

  // 近期比赛
  if (data.recent.length) {
    html += `<div class="card"><h3>近期比赛</h3><table><tr><th>时间</th><th>对阵</th><th>比分</th><th>结果</th></tr>`;
    data.recent.forEach(m => {
      html += `<tr><td>${(m.start_time||'').substr(0,10)}</td><td>${m.team_a} vs ${m.team_b}</td>
        <td>${m.team_a_score}:${m.team_b_score}</td>
        <td class="${m.our_win?'green':'red'}">${m.our_win?'胜':'负'}</td></tr>`;
    });
    html += '</table></div>';
  }

  document.getElementById('teamContent').innerHTML = html;
}

// ==================== 英雄分析 ====================

async function loadHeroSelect() {
  if (document.getElementById('heroSelect').options.length > 1) return;
}

async function loadHero() {
  const hid = document.getElementById('heroSelect').value;
  if (!hid) return;
  document.getElementById('heroContent').innerHTML = '<div class="loading">加载中...</div>';

  const data = await api('/api/hero/' + hid);
  const h = data.info;
  let html = `<div class="card"><h3>${h.hero_name}</h3></div>`;

  // 赛季趋势图
  if (data.season_stats.length) {
    html += `<div class="row">
      <div class="col-2"><div class="card"><h3>赛季趋势</h3><div class="chart-box short"><canvas id="heroTrendChart"></canvas></div></div></div>
      <div class="col"><div class="card"><h3>赛季数据</h3><table><tr><th>赛季</th><th>出场</th><th>Ban</th><th>胜率</th><th>KDA</th></tr>`;
    data.season_stats.forEach(s => {
      html += `<tr><td>${s.league_name}</td><td>${s.pick_count||0}</td><td>${s.ban_count||0}</td>
        <td>${((s.win_rate||0)*100).toFixed(0)}%</td><td>${s.avg_kda||'--'}</td></tr>`;
    });
    html += '</table></div></div></div>';
  }

  html += '<div class="row">';
  // 最佳搭档
  if (data.synergy.length) {
    html += `<div class="col"><div class="card"><h3>最佳搭档</h3>`;
    data.synergy.forEach(s => {
      const wr = Math.min(((s.win_rate || 0) * 100), 100).toFixed(0);
      html += `<div class="bar-row"><span style="width:80px">${s.hero_name}</span>
        <div style="flex:1;background:#1a1f3e;border-radius:3px">
          <div class="bar-fill bar-green" style="width:${wr}%"></div></div>
        <span>${wr}%</span><span class="small">${s.games_together}场</span></div>`;
    });
    html += '</div></div>';
  }
  // 最克制
  if (data.counter_win.length) {
    html += `<div class="col"><div class="card"><h3>最克制</h3>`;
    data.counter_win.forEach(c => {
      const wr = Math.min(((c.win_rate_a || 0) * 100), 100).toFixed(0);
      html += `<div class="bar-row"><span style="width:80px">${c.hero_name}</span>
        <div style="flex:1;background:#1a1f3e;border-radius:3px">
          <div class="bar-fill bar-green" style="width:${wr}%"></div></div>
        <span>${wr}%</span><span class="small">${c.games_against}场</span></div>`;
    });
    html += '</div></div>';
  }
  // 被克制
  if (data.counter_lose.length) {
    html += `<div class="col"><div class="card"><h3>被克制</h3>`;
    data.counter_lose.forEach(c => {
      const wr = Math.min(((c.win_rate_a || 0) * 100), 100).toFixed(0);
      html += `<div class="bar-row"><span style="width:80px">${c.hero_name}</span>
        <div style="flex:1;background:#1a1f3e;border-radius:3px">
          <div class="bar-fill bar-red" style="width:${wr}%"></div></div>
        <span>${wr}%</span><span class="small">${c.games_against}场</span></div>`;
    });
    html += '</div></div>';
  }
  html += '</div>';

  document.getElementById('heroContent').innerHTML = html;

  // 画趋势图
  setTimeout(drawHeroTrend, 100);
}

function drawHeroTrend() {
  const canvas = document.getElementById('heroTrendChart');
  if (!canvas) return;
  if (heroTrendChart) heroTrendChart.destroy();

  // 从英雄数据的table里提取赛季胜率
  const rows = document.querySelectorAll('#heroContent table tr');
  const labels = [], pickData = [], wrData = [];
  rows.forEach((row, i) => {
    if (i === 0) return; // skip header
    const cells = row.querySelectorAll('td');
    if (cells.length >= 4) {
      labels.unshift(cells[0].textContent.replace('年','').replace('KPL','').replace('王者荣耀','').slice(0,8));
      pickData.unshift(parseInt(cells[1].textContent) || 0);
      wrData.unshift(parseFloat(cells[3].textContent) || 0);
    }
  });

  heroTrendChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: '胜率%',
          data: wrData,
          borderColor: '#e6b422',
          backgroundColor: 'rgba(230,180,34,0.1)',
          yAxisID: 'y',
          tension: 0.3,
        },
        {
          label: '出场数',
          data: pickData,
          borderColor: '#2196f3',
          backgroundColor: 'rgba(33,150,243,0.1)',
          yAxisID: 'y1',
          tension: 0.3,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { position: 'left', grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: '#aaa' } },
        y1: { position: 'right', grid: { display: false }, ticks: { color: '#aaa' } },
        x: { ticks: { color: '#aaa', font: { size: 10 } } }
      },
      plugins: { legend: { labels: { color: '#aaa' } } }
    }
  });
}

// ==================== 启动 ====================

init();

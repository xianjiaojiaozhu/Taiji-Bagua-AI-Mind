// ============================================================
// 思维弦 · 卦锚记忆 数据模型层（含降级裁决流水线）
// ------------------------------------------------------------
// 概念映射（对齐《思维弦·宇宙轮回迭代理论》）：
//   爻        = 振动节点（弦振动的一段，卦位3爻=局部模态簇的三段振动）
//   weight(w) = 共振强度 / 观测激活度（0~1）
//   litLevel  = 点亮程度（视觉发光度，由共振强度驱动缓动，非瞬变）
//   base      = 本征底色（模态簇底色，演化只发生在表层附加振动，底色难抹除）
//   四象      = 全局共振场：控制所有爻权重的演化方向与速率
//               少阳=共振初起(生) · 太阳=共振极盛(盛) · 少阴=排斥初起(收) · 太阴=排斥极盛(藏)
//   reso      = 当前共振场强度：共振→权重上移→爻点亮；排斥→权重下沉→爻熄灭
//   事件      = 观测/交互：feed() 注入共振/排斥事件，直接驱动个别节点权重跳变
//   四卦流水线(降级) = 艮判→巽查→震降→兑归→坎位案卷；后端离线时本地自洽运转
//               本地无真实记忆网，集簇群权重用爻活性均值替代，巽查退化为活性判定
// ============================================================
const LAYER_R=[0.28,0.44,0.62,0.84]; // 四层半径
const N=8;
const GUATAGS=['乾','巽','坎','艮','坤','震','离','兑'];
const EMOTIONS={
  shaoyang:{a:'#3affd9', b:'#2bd88e', label:'少阳', px:-1, py:0, reso:0.55, drive:0.5,  litThr:0.50, fold:1.0, phase:'生', note:'共振初起 · 渐次点亮'},
  taiyang :{a:'#ffb84d', b:'#ff3d5a', label:'太阳', px:0,  py:1, reso:0.95, drive:1.4,  litThr:0.32, fold:1.7, phase:'盛', note:'共振极盛 · 大亮点亮'},
  shaoyin :{a:'#ffe8a0', b:'#ffab5e', label:'少阴', px:1,  py:0, reso:0.45, drive:0.5,  litThr:0.50, fold:1.0, phase:'收', note:'排斥初起 · 渐次熄灭'},
  taiyin  :{a:'#a48cff', b:'#5a4dff', label:'太阴', px:0,  py:-1,reso:0.12, drive:0.28, litThr:0.72, fold:0.6, phase:'藏', note:'排斥极盛 · 潜伏沉底'},
};
const clamp=(v,a,b)=>v<a?a:v>b?b:v;
const STORE_KEY='taiji_model_v1';
// ---- 线三：五行情绪层（空间轴） + 四相=层（时间轴） ----
const EMO5=['喜','怒','哀','乐','怨'];
const EMO5_SPAN={  // 五行 → 落宫卦位（idx = 层li×8 + 宫i；宫: 乾0 巽1 坎2 艮3 坤4 震5 离6 兑7）
  '喜':[6,14,22,30], '怒':[5,13,21,29,1,9,17,25], '哀':[0,8,16,24,7,15,23,31],
  '乐':[2,10,18,26], '怨':[4,12,20,28,3,11,19,27],
};
const EMO5_COLOR={'喜':'#ff5a5a','怒':'#3ee6a0','哀':'#ffe9a8','乐':'#4aa8ff','怨':'#d8b35a'};
const PHASES=[  // 四相 = 层：L0春少阳生爻(青) / L1夏太阳鼎爻(橙红) / L2秋少阴肃爻(金黄) / L3冬太阴蛰爻(紫)
  {layer:0, season:'春', xiang:'少阳', yi:'生爻', color:'#2bd88e'},
  {layer:1, season:'夏', xiang:'太阳', yi:'鼎爻', color:'#ff8a4d'},
  {layer:2, season:'秋', xiang:'少阴', yi:'肃爻', color:'#ffe9a8'},
  {layer:3, season:'冬', xiang:'太阴', yi:'蛰爻', color:'#a48cff'},
];
// 势定相（与后端对齐）：相层由乾势分档，非时间轴。qian≥0.75太阳L1夏 / ≥0.5少阳L0春 / ≥0.25少阴L2秋 / <0.25太阴L3冬；沿势序渐进滞回
const PHASE_XIANG=['shaoyang','taiyang','shaoyin','taiyin'];
const PHASE_SEASON=['春','夏','秋','冬'];
const PHASE_ORDER=[1,0,2,3];   // 势序：太阳>少阳>少阴>太阴；升/降势每轮最多一档（防阈值抖动跳相）
const EMO5_SEASON={'shaoyang':'怒','taiyang':'喜','shaoyin':'哀','taiyin':'乐'};

// 四卦流水线常量（与后端 server.py 对齐；LOCAL_ 前缀避免与 index.html 全局重名）
const LOCAL_GROUPS=[]; for(const li of [0,2]) for(let i=0;i<N;i++) LOCAL_GROUPS.push([li*N+i,(li+1)*N+i]);
const GENG_P=[3,11,19,27];  // 艮·决策
const XUN_P=[1,9,17,25];    // 巽·查因果
const DUI_P=[7,15,23,31];   // 兑·归盘
const ZHEN_P=[5,13,21,29];  // 震·降维
const V_HIGH=0.6, V_LOW=0.3, V_REPEL=0.35;
const WEAK_SECONDS=2*3600;   // 震卦后遗症：虚弱期2小时（与后端对齐）
const DEF_PARS={high:0.6, low:0.3, warn:60, rebound:0.8, weak:7200};  // 裁决阈值（可调）

const MODEL={
  emotion:'shaoyang',
  reso:0.55,        // 当前共振场强度（随四象演化）
  nodes:[],         // 爻节点（32卦位 × 3爻 = 96 振动节点）
  lastEvent:null,   // 最近注入的事件
  guarded:[],       // 高权重缓降卦位（艮判结果）
  verdict:{active:0,xun:0,dui:0,zhen:0,round:0,guarded:[],broken:[],saved:[],sunk:[],pending:[]},
  caseRecords:[],   // 坎位案卷（不可磨灭 · 本地持久化）
  caseViewLog:[],   // 坎位调阅日志：艮请求查历史 → 坎批准留痕（仅坎可查）
  weakUntil:{},     // 震卦后遗症：果端卦位 → 虚弱截止时间戳
  warnFlags:{},     // 三态机：卦位 → {st:1, w_orig, until}（State1 警告观察）
  pars:Object.assign({},DEF_PARS),   // 裁决阈值（实时可调）
  emTrans:null,      // 四象流转渐变：[from参数, t0]，1.5s 平滑过渡
  emo5W:{'喜':0,'怒':0,'哀':0,'乐':0,'怨':0},   // 五行情绪层权重（判官识别 → 累积 → 滑动衰减）
  phaseIdx:0,        // 势定相：当前相（0春/1夏/2秋/3冬，滞回相邻移动）
  phaseQian:0.5,     // 最近一次乾势快照（供本地降级显示）

  phaseFromQian(qian){
    // 乾势 → 四相档（势序渐进滞回：每轮最多一档，防阈值抖动跳相）。本地降级用，在线以 state.phase 为准
    const raw = qian>=0.75?1 : qian>=0.5?0 : qian>=0.25?2 : 3;
    let cp = PHASE_ORDER.indexOf(this.phaseIdx||0);
    const rp = PHASE_ORDER.indexOf(raw);
    if(rp<cp) this.phaseIdx=PHASE_ORDER[cp-1];      // 势升一档
    else if(rp>cp) this.phaseIdx=PHASE_ORDER[cp+1]; // 势降一档
    this.phaseQian=qian;
    return this.phaseIdx;
  },

  init(){
    this.nodes=[];
    for(let li=0;li<LAYER_R.length;li++){
      for(let i=0;i<N;i++){
        const base=0.25+Math.random()*0.5;      // 本征底色 0.25~0.75
        const yis=[];
        for(let k=0;k<3;k++){
          const yang=Math.random()<0.5;
          yis.push({yang, w:base+(Math.random()-0.5)*0.12, baseY:base, litLevel:0, emoLit:0});
        }
        this.nodes.push({layer:li,R:LAYER_R[li],angle:i*(360/N)-90,base,yis});
      }
    }
  },

  setEmotion(k){                       // 兼容旧调用：单情绪
    this.setEmotionW({[k]:1});
  },
  setEmotionW(w){                      // 相互共担：四象权重向量混合（悲喜交加=太阳0.5+太阴0.5）
    const keys=['shaoyang','taiyang','shaoyin','taiyin'];
    const nw={}; let tot=0;
    for(const k of keys){ nw[k]=w[k]||0; tot+=nw[k]; }
    if(tot<=0) return;
    for(const k of keys) nw[k]/=tot;   // 归一化
    const cur=this.emotionParams();    // 渐变起点 = 当前实际参数（含未完成的过渡）
    this.emTrans=[cur, Date.now()/1000];
    this.emoW=nw;
    this.emotion=this.emoLabel(nw);
  },
  emoLabel(w){
    const parts=[];
    for(const k of Object.keys(EMOTIONS)) if((w[k]||0)>0.05) parts.push(EMOTIONS[k].label);
    return parts.join('·')||'少阳';
  },
  emoSynth(){                          // 四象权重合成：动力学参数凸组合（混合情绪 = 共享场共担）
    const w=this.emoW||{shaoyang:1};
    const out={reso:0,drive:0,litThr:0,fold:0};
    for(const k of Object.keys(EMOTIONS)){
      const e=EMOTIONS[k], ww=w[k]||0;
      out.reso+=e.reso*ww; out.drive+=e.drive*ww; out.litThr+=e.litThr*ww; out.fold+=e.fold*ww;
    }
    return out;
  },
  emoColA(){                           // 混合情绪主色（a 端按权重 mix）
    const w=this.emoW||{shaoyang:1};
    let r=0,gg=0,b=0;
    for(const k of Object.keys(EMOTIONS)){
      const e=EMOTIONS[k], ww=w[k]||0;
      r+=parseInt(e.a.slice(1,3),16)*ww; gg+=parseInt(e.a.slice(3,5),16)*ww; b+=parseInt(e.a.slice(5,7),16)*ww;
    }
    return '#'+((r<<16)|(gg<<8)|b).toString(16).padStart(6,'0');
  },
  emotionParams(){                     // 当前实际动力学参数（渐变中值，混合合成）
    if(this.emTrans){
      const f=this.emTrans[0], t0=this.emTrans[1];
      const p=Math.min(1,(Date.now()/1000-t0)/1.5);
      if(p>=1){ this.emTrans=null; return this.emoSynth(); }
      const tar=this.emoSynth();
      return {reso:f.reso+(tar.reso-f.reso)*p, drive:f.drive+(tar.drive-f.drive)*p,
              litThr:f.litThr+(tar.litThr-f.litThr)*p, fold:f.fold+(tar.fold-f.fold)*p};
    }
    return this.emoSynth();
  },

  // 演化一帧：共振场收敛 + 每爻权重向"场强+底色偏移"的目标收敛 + 表层振动噪声
  tick(dt){
    const e=this.emotionParams();                // 渐变中的动力学参数
    this.reso += (e.reso-this.reso)*Math.min(1,e.drive*dt*0.8);
    const r=this.reso;
    const guardedSet=new Set(this.guarded);
    const now=Date.now()/1000;
    for(let idx=0;idx<this.nodes.length;idx++){
      const nd=this.nodes[idx];
      const guard=guardedSet.has(idx);
      const weak=this.weakUntil[idx]>now;                 // 震后遗症：虚弱期
      for(const yi of nd.yis){
        let target=clamp(r+(nd.base-0.5)*0.6,0,1);
        if(weak) target=Math.min(target,0.18);            // 虚弱期：太极场静默
        yi.w += (target-yi.w)*Math.min(1,e.drive*dt*0.8) + (Math.random()-0.5)*0.002*e.fold;
        yi.w=clamp(yi.w,0,1);
        const tgt=yi.w>e.litThr?1:0;
        let rate=e.fold>1?2.2:(e.fold<1?0.6:1.2);
        if(guard && tgt===0) rate*=0.3;              // 高权重缓降：灭得慢，留余温
        yi.litLevel += (tgt-yi.litLevel)*Math.min(1,rate*dt);
      }
    }
    // ---- 五行情绪层（空间轴）：落宫卦位按权重档分层加亮（emoLit 独立，两场解耦） ----
    for(const emo of EMO5){
      const w5=this.emo5W[emo]||0;
      const layers= w5>0.01 ? (1+(w5>0.25)+(w5>0.5)+(w5>0.75)) : 0;
      const tgt=0.55+0.3*Math.min(1,w5);
      for(const ndi of EMO5_SPAN[emo]){
        const nd=this.nodes[ndi];
        for(const yi of nd.yis){
          const cur=yi.emoLit||0;
          const want=(w5>0.01 && nd.layer<layers)?tgt:0;
          if(Math.abs(cur-want)>0.001) yi.emoLit+=(want-cur)*Math.min(1,0.25*dt);
        }
      }
    }
    // 五行情绪滑动衰减（秒级 ~1.5%/s，情绪自然消退）
    const f=Math.pow(0.985,dt);
    for(const k of EMO5) this.emo5W[k]=Math.max(0,(this.emo5W[k]||0)*f);
  },

  // 五行情绪入层：判官识别情绪 → 权重累积（季节当令加成），随 tick 滑动衰减
  addEmo5(emo,conf,seasonBoost){
    if(!EMO5.includes(emo)) return;
    this.emo5W[emo]=Math.min(2,(this.emo5W[emo]||0)+0.6+0.4*conf+(seasonBoost||0));
  },

  // 事件注入：idx=节点索引, type='resonance'(共振,亮) / 'repel'(排斥,灭), strength=强度0~1
  feed(idx,type,strength){
    const nd=this.nodes[idx]; if(!nd)return;
    if(type==='resonance' && this.weakUntil[idx]>Date.now()/1000) strength=0;  // 虚弱期：骂不涨
    for(const yi of nd.yis){
      yi.w=clamp(yi.w+(type==='resonance'?1:-1)*strength,0,1);
    }
    this.lastEvent={idx,type,str:strength, t:performance.now()};
  },

  // 自动事件：按当前四象混合权重随机选节点注入（阳系权重>0.5 → 共振，否则排斥）
  autoEvent(){
    const w=this.emoW||{shaoyang:1};
    const idx=(Math.random()*this.nodes.length)|0;
    const type=(w.shaoyang+w.taiyang)>0.5?'resonance':'repel';
    const str=0.15+Math.random()*0.18;
    this.feed(idx,type,str);
    return this.lastEvent;
  },

  // 当前点亮节点数（共振灯 或 五行情绪灯，任一>0.5）
  litCount(){
    let c=0; for(const nd of this.nodes) for(const yi of nd.yis) if(Math.max(yi.litLevel,yi.emoLit||0)>0.5)c++;
    return c;
  },

  // ---- 降级裁决：四卦流水线（艮判→巽查→震降→兑归→坎位案卷）----
  avgPill(ids){ let s=0; for(const i of ids) for(const y of this.nodes[i].yis) s+=y.litLevel; return s/(ids.length*3); },
  guaLit(idx){ const nd=this.nodes[idx]; return nd?nd.yis.reduce((a,y)=>a+y.litLevel,0)/3:0.5; },
  caseLog(cut, keep){
    const tag=GUATAGS[cut%8], layer=cut>>3, keepTag=GUATAGS[keep%8];
    const d=new Date();
    const ds=`${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
    const text=`[${ds}] 艮判：${tag}L${layer}恶爻高危。巽查：因果确凿，因是${keepTag}。震：已降维封存。留存100天。坎位档案`;
    this.caseRecords.push({ts:d.getTime()/1000, text, keep_days:100, expire:d.getTime()/1000+100*86400, target:cut});
    if(this.caseRecords.length>200) this.caseRecords=this.caseRecords.slice(-200);
  },
  caseGroupLog(cuts){
    const groups={};
    for(const c of cuts){ const key=GUATAGS[c%8]+'L'+(c>>3); (groups[key]=groups[key]||[]).push(c); }
    const d=new Date();
    const ds=`${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
    for(const key in groups){
      const desc=groups[key].map(c=>`${GUATAGS[c%8]}L${c>>3}`).join('、');
      const text=`[${ds}] 艮判：${desc}恶爻高危。巽查：因果确凿，同源因是${key}。震：已降维封存。留存100天。坎位档案`;
      const members=groups[key];
      this.caseRecords.push({ts:d.getTime()/1000, text, keep_days:100, expire:d.getTime()/1000+100*86400,
                             target:members[0], targets:members});   // 与后端对齐：统计口径含全部果端
    }
    if(this.caseRecords.length>200) this.caseRecords=this.caseRecords.slice(-200);
  },
  weakRemain(){ const now=Date.now()/1000; const out={}; for(const k in this.weakUntil){ if(this.weakUntil[k]>now) out[k]=Math.round(this.weakUntil[k]-now); } return out; },
  caseStats(){ const m={}; for(const r of this.caseRecords){ const ts=r.targets&&r.targets.length?r.targets:[r.target]; for(const t of ts){ if(t!==undefined&&t!==null) m[t]=(m[t]||0)+1; } } return m; },
  warnRemain(){ const now=Date.now()/1000; const out={}; for(const k in this.warnFlags){ const f=this.warnFlags[k]; if(f.st===1 && f.until>now) out[k]=Math.round(f.until-now); } return out; },
  guaW(idx){ const nd=this.nodes[idx]; return nd?nd.yis.reduce((a,y)=>a+y.w,0)/3:0.5; },
  cutYao(idx){   // State2 Purge：断爻处决
    for(const yi of this.nodes[idx].yis){ yi.w=0.05; yi.litLevel=0; }
    this.weakUntil[idx]=Date.now()/1000+this.pars.weak;
    delete this.warnFlags[idx];
  },
  applyLocalParams(kv){   // 本地降级调参：{key?, reso?, drive?, litThr?, fold?, high?, low?, warn?, rebound?, weak?}
    if(kv.key && EMOTIONS[kv.key]){
      const e=EMOTIONS[kv.key];
      for(const k of ['reso','drive','litThr','fold']) if(kv[k]!==undefined) e[k]=kv[k];
    }
    for(const k of ['high','low','warn','rebound','weak']) if(kv[k]!==undefined) this.pars[k]=kv[k];
    this.save();
  },
  verdictTick(){
    const s=this.avgPill(GENG_P), sx=this.avgPill(XUN_P), sd=this.avgPill(DUI_P), sz=this.avgPill(ZHEN_P);
    const guarded=[],broken=[],saved=[],sunk=[],pending=[];
    const brokenReasons={};
    const now=Date.now()/1000;
    for(const g of LOCAL_GROUPS){
      const a=g[0],b=g[1];
      const wa=this.guaLit(a), wb=this.guaLit(b);
      const wg=Math.min(wa,wb);
      if(wg>=this.pars.high){ guarded.push(a,b); continue; }
      if(wg<this.pars.low && s>0.05){
        if(sx<=0.05||sz<=0.05||sd<=0.05){ pending.push([a,b]); continue; }  // 流水线缺位→挂起
        const cut=wa<=wb?a:b, keep=cut===a?b:a;                              // 巽查(退化)：活性低端为果
        const now=Date.now()/1000;
        if(this.weakUntil[cut]>now) continue;                                // 虚弱期：不再重复裁决
        // 三态机：State1 警告观察 → 反弹则 State2 处决，否则释放（艮先请坎查历史）
        const flag=this.warnFlags[cut];
        if(!flag){
          const prior=this.caseRecords.filter(r=>r.target===cut).length;     // 坎查历史：前科数
          if(prior>=2){                                                      // 惯犯：直接处决
            this.cutYao(cut); broken.push(cut); brokenReasons[cut]='惯犯';
            this.caseViewLog.push({ts:now, idx:cut, n:prior, mode:'惯犯·直判'});
            continue;
          }
          const w_orig=this.guaW(cut);
          this.feed(cut,'repel', s*sz*V_REPEL);                              // 触发排斥
          for(const yi of this.nodes[cut].yis) yi.w=clamp(yi.w*0.5,0,1);     // 权重×0.5 + [震卦警告]
          const until=now+(prior?this.pars.warn/2:this.pars.warn);           // 重犯观察期减半
          this.warnFlags[cut]={st:1, w_orig, until, prior};
          if(prior) this.caseViewLog.push({ts:now, idx:cut, n:prior, mode:'重犯·观察减半'});
        } else if(flag.st===1 && now>=flag.until){
          if(this.guaW(cut) >= this.pars.rebound*flag.w_orig){               // 恶性反弹 → Purge
            this.cutYao(cut); broken.push(cut); brokenReasons[cut]='反弹';
          } else {
            delete this.warnFlags[cut];                                      // 悔改释放
          }
        }        const wd=this.guaLit(keep);
        if(wd>=0.5) saved.push(keep); else sunk.push(keep);                  // 兑归盘
      }
    }
    if(broken.length) this.caseGroupLog(broken);                             // 同源合并立档
    this.guarded=guarded;
    this.verdict={active:s,xun:sx,dui:sd,zhen:sz,round:this.verdict.round+1,
                  guarded,broken,saved,sunk,pending,brokenReasons,
                  chainInfo:{}};   // 本地降级无因果网 → 跨卦连锁由后端判定（chainInfo 在线填充）
  },

  // 持久化（记忆不因页面关闭而丢失）
  save(){
    try{
      localStorage.setItem(STORE_KEY, JSON.stringify({
        emotion:this.emotion, reso:this.reso,
        nodes:this.nodes.map(nd=>({layer:nd.layer,R:nd.R,angle:nd.angle,base:nd.base,
          yis:nd.yis.map(y=>({yang:y.yang,w:y.w,baseY:y.baseY,litLevel:y.litLevel,emoLit:y.emoLit||0}))})),
        guarded:this.guarded, verdict:this.verdict, caseRecords:this.caseRecords,
        weakUntil:this.weakUntil, warnFlags:this.warnFlags, pars:this.pars,
        caseViewLog:this.caseViewLog, emo5W:this.emo5W,
        phaseIdx:this.phaseIdx, phaseQian:this.phaseQian
      }));
    }catch(e){}
  },
  load(){
    try{
      const d=JSON.parse(localStorage.getItem(STORE_KEY));
      if(d && d.nodes && d.nodes.length===this.nodes.length){
        this.emotion=d.emotion; this.reso=d.reso; this.nodes=d.nodes;
        this.guarded=d.guarded||[]; this.verdict=d.verdict||this.verdict;
        this.caseRecords=d.caseRecords||[];
        this.caseViewLog=d.caseViewLog||[];
        this.weakUntil=d.weakUntil||{};
        this.warnFlags=d.warnFlags||{};
        if(d.pars) this.pars=Object.assign({},DEF_PARS,d.pars);
        if(d.emo5W){ const nw={}; for(const k of EMO5) nw[k]=Number(d.emo5W[k]||0); this.emo5W=nw; }
        if(typeof d.phaseIdx==='number') this.phaseIdx=d.phaseIdx;
        if(typeof d.phaseQian==='number') this.phaseQian=d.phaseQian;
        return true;
      }
    }catch(e){}
    return false;
  },
};

MODEL.init();

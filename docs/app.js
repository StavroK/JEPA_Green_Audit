const zones = [
  {name:"Monterrey Central Pilot",before:24.8,after:21.6,jepa:0.73,seed:7},
  {name:"Park Corridor Pilot",before:38.4,after:39.7,jepa:0.18,seed:11},
  {name:"Industrial Edge Pilot",before:17.2,after:12.9,jepa:0.86,seed:23},
  {name:"Dense Urban Block Pilot",before:9.7,after:9.1,jepa:0.31,seed:31}
];

const zoneSelect=document.getElementById("zoneSelect");
zones.forEach((z,i)=>{const o=document.createElement("option");o.value=i;o.textContent=z.name;zoneSelect.appendChild(o)});

function clamp(v,min,max){return Math.max(min,Math.min(max,v))}
function seeded(seed,i){const x=Math.sin(seed*997+i*131)*43758.5453;return x-Math.floor(x)}

function renderGrid(zone){
  const map=document.getElementById("auditMap");map.innerHTML="";
  for(let i=0;i<48;i++){
    const r=seeded(zone.seed,i);
    const base=Math.max(0,zone.before-zone.after);
    const score=clamp(Math.round((r*.48+base/10*.52)*100),3,96);
    const cell=document.createElement("div");
    cell.className="cell "+(score>=65?"high":score>=35?"medium":"low");
    cell.dataset.score=score;
    cell.title="Prototype audit signal: "+score;
    map.appendChild(cell);
  }
}

function run(){
  const z=zones[Number(zoneSelect.value)];
  const delta=z.after-z.before;
  const loss=Math.max(0,-delta);
  const priority=Math.round(clamp((loss/10)*65+z.jpa*35,0,100));
  document.getElementById("coverage").textContent=z.after.toFixed(1)+"%";
  document.getElementById("change").textContent=(delta>0?"+":"")+delta.toFixed(1)+" pp";
  document.getElementById("jepa").textContent=z.jpa.toFixed(2);
  document.getElementById("priority").textContent=priority+"/100";
  document.getElementById("findingTitle").textContent=z.name;
  document.getElementById("beforeCoverage").textContent=z.before.toFixed(1)+"%";
  document.getElementById("afterCoverage").textContent=z.after.toFixed(1)+"%";
  document.getElementById("direction").textContent=delta>0?"Estimated gain":delta<0?"Estimated loss":"Stable";
  document.getElementById("explanation").textContent=
    delta<0
      ? `Sample coverage decreased by ${Math.abs(delta).toFixed(1)} percentage points while the latent-representation change signal is ${z.jpa.toFixed(2)}. The combination raises this zone for imagery review; it does not establish why vegetation changed.`
      : `Sample coverage is stable or improving. The latent-representation change signal is ${z.jpa.toFixed(2)}; review is lower priority unless source-image quality or local context suggests otherwise.`;
  document.getElementById("action").textContent=
    priority>=60
      ? "Inspect aligned before/after imagery, confirm cloud and seasonal quality, compare NDVI, and create a field-validation task if the change remains credible."
      : "Retain for routine monitoring and re-evaluate when the next observation is available.";
  renderGrid(z);
}

document.getElementById("runAudit").addEventListener("click",run);
zoneSelect.addEventListener("change",run);
run();

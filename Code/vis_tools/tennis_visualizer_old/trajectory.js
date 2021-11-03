var ball;

var tail;
var traj, traj_line;
var gt_traj, gt_traj_line;

function defaultBallColoring(p) {
  return new THREE.MeshStandardMaterial( { 
    color: new THREE.Color(1.0, 240/255.0, 0.0), 
    opacity: 0.8,
    transparent: true} );
}

function defaultLineColoring(p) {
  return new THREE.LineBasicMaterial( { color: new THREE.Color(1.0, 240/255.0, 0.0)} );
}

function addTrajectory(data, ball_color=defaultBallColoring, line_color=defaultLineColoring, gt_ball_color=defaultBallColoring, gt_line_color=defaultLineColoring) {

  function disposeTraj(t, tl) {
    scene.remove(t);
    scene.remove(tl);
    for (let i = 0; i < t.children.length; i++) {
      t.children[i].geometry.dispose();
      t.children[i].material.dispose();
      if (i < tl.children.length) {
        tl.children[i].geometry.dispose();
        tl.children[i].material.dispose();
      }
    }
  }
  if (typeof traj !== "undefined") 
    disposeTraj(traj, traj_line);
  if (typeof gt_traj !== "undefined") 
    disposeTraj(gt_traj, gt_traj_line);


  function addTraj(dat, t, tl, ballcolor, linecolor) {
    for (let i = 0; i < dat.length; i++) {
      const d = dat[i];
      const color1 = ballcolor(d);
      const color2 = linecolor(d);
      const geometry = new THREE.SphereGeometry(config.tballsize, 16, 16);

      const sphere = new THREE.Mesh(geometry, color1);
      sphere.position.set(d[0], d[1], -d[2]);
      sphere.castShadow = true; // Unity to GL
      sphere.visible = config.showall;
      t.add(sphere);

      if (i > 0) {
        const d1 = dat[i-1];
        const points = [];
        points.push( new THREE.Vector3(d[0], d[1], -d[2]));
        points.push( new THREE.Vector3(d1[0], d1[1], -d1[2]))

        const geometry = new THREE.BufferGeometry().setFromPoints( points );
        const line = new THREE.Line( geometry, color2);
        line.visible = config.showline;
        tl.add(line);
      }
    }
    scene.add(t);
    scene.add(tl);
  }

  if (data["pred"] != null && data["pred"].length > 0) {
    traj = new THREE.Group();
    traj_line = new THREE.Group();
    addTraj(data["pred"], traj, traj_line, ball_color, line_color);
  }

  if (data["gt"] != null && data["gt"].length > 0) {
    gt_traj = new THREE.Group();
    gt_traj_line = new THREE.Group();
    addTraj(data["gt"], gt_traj, gt_traj_line, gt_ball_color, gt_line_color);
  }


  addBall(scene);
  addTail2(0, traj.children[0], 0);
}

function addBall() {
  if (typeof ball !== 'undefined') {
    scene.remove(ball);
    ball.geometry.dispose();
    ball.material.dispose();
  }

  const geometry = new THREE.SphereGeometry(config.ballsize, 16, 16);
  const material = new THREE.MeshPhongMaterial( { 
    color: 0xc4c931,
    emissive: 0x3d3f13,
  } );
  ball = new THREE.Mesh( geometry, material );
  ball.castShadow = true; 
  scene.add(ball);
}

var tailMaterial = null;
var cat;

function addTail2(id, ball, t) {
  if (typeof tail !== "undefined") {
    scene.remove(tail);
    tail.geometry.dispose();
  }

  let helix = [ball.position];
  for (let i = 0; i < config.drawtail; i++) 
    helix.push(traj.children[Math.max(0, id - i - 1)].position);

  class MyCurve extends THREE.Curve {
    constructor() {
      super();
    }

    getPoint(t, optionalTarget = new THREE.Vector3() ) {
      let id = Math.round(t * config.drawtail);
      const point = optionalTarget;
      const pos = helix[id];
      //point.set(1.0 * Math.round(t * config.drawtail), 
        //Math.round(t * config.drawtail), t * config.drawtail);
      point.x = t * 10;
      point.y = Math.round(t * 10);
      point.z = Math.round(t * 10);
      return point;
    }
  };

  //var curve = new MyCurve(); //new THREE.CatmullRomCurve3( helix );
  var curve = new THREE.CatmullRomCurve3( helix );
  var geometry = new THREE.TubeGeometry(curve, config.drawtail * 4, config.ballsize * 0.7, 8, false);

  if (tailMaterial == null) {
    const ctx = document.createElement('canvas').getContext('2d');
    ctx.canvas.width = 2;//config.drawtail;
    ctx.canvas.height = 2;
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, 2, 2);
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, 1, 2);
    const map = new THREE.CanvasTexture(ctx.canvas);
    const color = new THREE.Color(1, 0.8, 0.1);
    tailMaterial = new THREE.MeshBasicMaterial( { 
      color: color, 
      side: THREE.DoubleSide,
      alphaMap: map,
      transparent: true,
      depthWrite: false,
    });
  }

  tail = new THREE.Mesh(geometry, tailMaterial);
  scene.add(tail);
}

function addTail(id, ball, t) {
  if (typeof tail !== "undefined") {
    scene.remove(tail);
    for (let i = 0; i < tail.children.length; i++) {
      tail.children[i].geometry.dispose();
      tail.children[i].material.dispose();
    }
  }

  tail = new THREE.Group();
  for (let i = 0; i < config.drawtail; i++) {
    let helix = [];
    let id0 = id - i; if (id0 < 0) id0 = 0;
    let id1 = id0 - 1; if (id1 < 0) id1 = 0;

    if (i == 0) 
      helix.push(ball.position.clone());
    else
      helix.push(traj.children[id0].position.clone());

    helix.push(traj.children[id1].position.clone());

    var curve = new THREE.CatmullRomCurve3( helix );
    var geometry = new THREE.TubeGeometry(curve, 1, config.ballsize * 0.7, 8, false);
    //let color = new THREE.Color(0.7, 1, 0.7);
    let color = new THREE.Color(1, 0.8, 0.1);
    const material = new THREE.MeshBasicMaterial( { 
      color: color, 
      side: THREE.DoubleSide, 
      transparent: true, 
      opacity: (1-i/config.drawtail) * (1-t) + (1-(i+1)/config.drawtail) * t,
      depthWrite: false,
    } );
    tail.add(new THREE.Mesh( geometry, material ));
  }
  scene.add(tail);
}


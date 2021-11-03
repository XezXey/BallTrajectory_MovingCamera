var anim = 0
var clock = new THREE.Clock();

var camera;
var renderer;
var vwidth;
var vheight;

var scene;
var cb;
var bound;
var debug = 0;

const config = {
  dirlightRadius: 4,
  dirlightSamples: 12,
  shadow: false,
  showall: true,
  showgt: true,
  showline: false,
  speed: 0.05,
  drawtail: 10,
  traj_id: 0,
  ballsize: 0.06,
  tballsize: 0.04,
  tballsize_base: 0.03,
  ballsize_base: 0.04,
  patch_size: 0.5,
  cb_size: 10,
};

const views = [
  {
    left: 0.3,
    bottom: 0,
    width: 0.7,
    height: 1.0,
    background: new THREE.Color( 0xFFFFFF ),
    eye: [ 0, 300, 1800 ],
    up: [ 0, 1, 0 ],
  },
  {
    left: 0,
    bottom: 2.0/3,
    width: 0.3,
    height: 1.0/3,
    background: new THREE.Color(0xeFefef),
    eye: [ 0, 2, 20 ],
    lookAt: [0, 2, -20],
    up: [ 0, 1, 0 ],
  },
  {
    left: 0,
    bottom: 1.0/3,
    width: 0.3,
    height: 1.0/3,
    background: new THREE.Color(0xe6e6e6),
    eye: [ 20, 2, 0 ],
    lookAt: [-20, 2, 0],
    up: [ 0, 1, 0 ],
  },
  {
    left: 0,
    bottom: 0,
    width: 0.3,
    height: 1.0/3,
    background: new THREE.Color(0xe6e6e6),
    eye: [ 0, 40, 0],
    lookAt: [0, 0, 0],
    up: [ 0, 0, -1 ],
  }
];

var dirLight1;

function drawFlatLine(scene, a, b, w, stand=0, color=0xffffff, castShadow=false) {
  if (Array.isArray(a)) 
    a = new THREE.Vector3(a[0], a[1], a[2]);
  if (Array.isArray(b)) 
    b = new THREE.Vector3(b[0], b[1], b[2]);

  if (stand)
    var up = new THREE.Vector3 (0.5, 0, 0.5);
  else
    var up = new THREE.Vector3 (0, 1, 0);

  let diff = b.clone(); diff.sub(a);
  let diff_u = diff.clone();
  diff_u.normalize();

  let orth = diff.clone();
  orth.cross(up);
  orth.normalize();

  let p = new Array(4);
  p[0] = b.clone();
  p[0].addScaledVector(orth, w / 2);
  p[0].addScaledVector(diff_u, w / 2); 
  p[1] = p[0].clone();
  p[1].addScaledVector(orth, -w);

  p[3] = a.clone();
  p[3].addScaledVector(orth, w / 2);
  p[3].addScaledVector(diff_u, -w / 2); 
  p[2] = p[3].clone();
  p[2].addScaledVector(orth, -w);

  //const material = new THREE.MeshNormalMaterial();
  const material = new THREE.MeshBasicMaterial( { color: color, side: THREE.DoubleSide} );
  let geometry = new THREE.BufferGeometry()
  const points = [p[0], p[1], p[2], p[0], p[2], p[3]];
  geometry.setFromPoints(points);
  geometry.computeVertexNormals();

  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = castShadow;
  scene.add(mesh);
}

function setupGUI() {
  const gui = new dat.GUI({width: 300, autoPlace: false});
  $("#control").prepend(gui.domElement);
  
  gui.useLocalStorage = true;

  const folder_traj = gui.addFolder( 'Trajectory' );
  folder_traj.add(config, 'traj_id', Object.keys(data)).name('Trajectory ID').listen().onChange(function(value) {
    console.log("Showing trajectory ", config.traj_id);
    customAddTrajectory();
  });

  folder_traj.add(config, 'showall').name('Show Prediction').listen().onChange( function(value) { 
    for (let i = 0; i < traj.children.length; i++) {
      traj.children[i].visible = value;
    }
  });
  folder_traj.add(config, 'showgt').name('Show Groundtruth').listen().onChange( function(value) { 
    for (let i = 0; i < gt_traj.children.length; i++) {
      gt_traj.children[i].visible = value;
    }
  });
  //folder_traj.add(config, 'showline').name('Show Line').listen().onChange( function(value) { 
    //for (let i = 0; i < traj_line.children.length; i++) {
      //traj_line.children[i].visible = value;
    //}
  //});
  folder_traj.add(config, 'ballsize_base', 0.01, 0.1, 0.01).name('Ball radius').listen().onChange(function(value) {
    customAddTrajectory();
  });
  folder_traj.add(config, 'tballsize_base', 0.01, 0.1, 0.01).name('Trajectory radius').listen().onChange(function(value) {
    customAddTrajectory();
  });

  folder_traj.add(config, 'drawtail', 0, 30, 1).name('Draw Tail').listen().onChange(function(value) {
    if (value == 0 && typeof tail !== 'undefined') {
      scene.remove(tail);
      for (let i = 0; i < tail.children.length; i++) {
        tail.children[i].geometry.dispose();
        tail.children[i].material.dispose();
      }
    }
  });

  folder_traj.add(config, 'speed', 0, 2, 0.05).name('Speed');

  const folder_render = gui.addFolder( 'Rendering' );
  folder_render.add(config, 'shadow').name('Shadow').listen().onChange( function(value) {
    renderer.shadowMap.enabled = value;
    scene.traverse(function (child) {
      if (child.material) {
        child.material.needsUpdate = true
      }
    })
  });

  const dirlightFolder = folder_render.addFolder( 'Directional Light' );
  dirlightFolder.add( config, 'dirlightRadius', 0, 25, 0.1).name( 'Shadow Radius' ).listen().onChange( function ( value ) {
    dirLight1.shadow.radius = value;
  } );

  dirlightFolder.add( config, 'dirlightSamples', 1, 64, 1).name( 'Shadow Samples' ).listen().onChange( function ( value ) {
    dirLight1.shadow.blurSamples = value;
  } );

  folder_traj.open();
  folder_render.open();
  //dirlightFolder.open();

}

// size = metric size of one patch
function addCheckerboard(patch_size, size) {
  let rep = Math.ceil(size / patch_size);
  
  console.log(rep, patch_size);
  var geom = new THREE.PlaneGeometry(rep * patch_size, rep * patch_size, rep, rep).toNonIndexed();
  geom.rotateX(-0.5*Math.PI);

  const ctx = document.createElement('canvas').getContext('2d');
  ctx.canvas.width = 2;
  ctx.canvas.height = 2;
  ctx.fillStyle = '#a6a6a6';
  ctx.fillRect(0, 0, 2, 2);
  ctx.fillStyle = '#6c6c6c';
  ctx.fillRect(0, 1, 1, 1);
  const texture = new THREE.CanvasTexture(ctx.canvas);
  texture.magFilter = THREE.NearestFilter;
  const material = new THREE.MeshPhongMaterial( { 
    color: 0xffffff,
    map: texture, 
    opacity: 0.8, 
    transparent: true});

  const uv = geom.attributes.uv;
  let counter = 0, flip = 0;
  for (let i = 0; i < uv.count; i++) {
    if (i > 0 && i % 6 == 0) {
      counter ++;
      if (counter % rep == 0) {
        flip = 1 - flip;
      }
    }
    uv.setXY(i, (counter+flip) % 2, (counter+flip) % 2);
  }
  var checkercolor = new THREE.Mesh(geom, material);
  checkercolor.receiveShadow = true;

  var geom2 = new THREE.PlaneGeometry(rep * patch_size, rep * patch_size, rep, rep);
  var groundMirror = new THREE.Reflector(geom2, {
    clipBias: 0.003,
    textureWidth: vwidth * (views[0].right - views[0].bottom), //window.innerWidth * window.devicePixelRatio,
    textureHeight: vheight * (views[0].bottom - views[0].top), //window.innerHeight * window.devicePixelRatio,
    patch_size: config.patch_size

  });
  groundMirror.rotateX(-Math.PI / 2);
  groundMirror.position.y = -0.001;
  groundMirror.receiveShadow = true;
  
  //cb.renderOrder = 2;
  //groundMirror.renderOrder = 1;
  cb = new THREE.Group();
  cb.add(groundMirror);
  cb.add(checkercolor);
  scene.add(cb);
}

function blueColoring(point, simple=false) {
  const int = 0.3;
  const em = 0.7;
  const color = [43 / 255, 185 / 255, 204 / 255];
  //const color = [48 / 255, 143 / 255, 210 / 255];
  
  if (simple) 
    return new THREE.MeshBasicMaterial({color: 0x18acbf});
  return new THREE.MeshPhongMaterial( { 
    color: new THREE.Color(color[0] * int, color[1] * int, color[2] * int), 
    emissive: new THREE.Color(color[0] * em, color[1] * em, color[2] * em), 
    specular: new THREE.Color(0.1, 0.1, 0.1), 
    opacity: 1,
    shininess: 5,
    transparent: true} );
}

function redColoring(point, simple=false) {
  const int = 0.3;
  const em = 0.8;
  const color = [180 / 255, 27 / 255, 33 / 255];
  if (simple)
    return new THREE.MeshBasicMaterial({color: 0xca1e26});
  return new THREE.MeshPhongMaterial( { 
    color: new THREE.Color(color[0] * int, color[1] * int, color[2] * int), 
    emissive: new THREE.Color(color[0] * em, color[1] * em, color[2] * em), 
    specular: new THREE.Color(0.1, 0.1, 0.1), 
    opacity: 1,
    shininess: 5,
    transparent: true} );
}

function customAddTrajectory() {
  window.location.hash = config.traj_id;
  analyzeTrajectory(data[config.traj_id]);
  addTrajectory(data[config.traj_id], blueColoring, defaultLineColoring, redColoring, defaultLineColoring);
}

function analyzeTrajectory(data) {

  function getPrincipleDirection(d) {
    let newlist = [];
    let mean = [0, 0, 0]; 
    for (let i = 0; i < d.length; i++) {
      for (let j = 0; j < 3; j++) {
        mean[j] += d[i][j];
      }
    }
    for (let j = 0; j < 3; j++)
      mean[j] /= d.length;
    for (let i = 0; i < d.length; i++) {
      newlist.push([d[i][0] - mean[0], d[i][1] - mean[1], d[i][2] - mean[2]]);
    }
    var vectors = PCA.getEigenVectors(newlist);
    return vectors[0].vector;
  }

  function fb(d) {
    mean = [0, 0, 0]; 
    for (let i = 0; i < d.length; i++) {
      for (let j = 0; j < 3; j++) {
        bound[j][0] = Math.min(bound[j][0], d[i][j]);
        bound[j][1] = Math.max(bound[j][1], d[i][j]);
      }
    }
  }

  function centering(d) {
    let mid = [0, 0, 0];
    for (let i = 0; i < 3; i++) {
      mid[i] = (bound[i][0] + bound[i][1]) / 2;
    }
    for (let i = 0; i < d.length; i++) {
      for (let j = 0; j < 3; j+=2) {
        d[i][j] -= mid[j];
      }
    }
  }

  function rotating(d, singvec) {
    let xz_len = Math.sqrt(singvec[0] * singvec[0] + singvec[2] * singvec[2]);
    let x = singvec[0] / xz_len;
    let z = singvec[2] / xz_len;

    console.log(x, z);
    for (let i = 0; i < d.length; i++) {
      let ox = d[i][0], oz = d[i][2];
      d[i][0] = x * ox + z * oz;
      d[i][2] = -z * ox + x * oz;
    }
  }


  let svec = getPrincipleDirection(data["pred"]);
  console.log(svec);

  rotating(data["pred"], svec);
  rotating(data["gt"], svec);

  bound = [[1e10, -1e10], [1e10, -1e10], [1e10, -1e10]];
  fb(data["pred"]);
  fb(data["gt"]);

  centering(data["pred"]);
  centering(data["gt"]);

  let heights = [0, 0];
  const d = data["pred"];
  for (let i = 0; i < d.length; i++) 
    heights[d[i][0] < 0 ? 0 : 1] += d[i][1];
  if (heights[1] > heights[0]) {
    rotating(data["pred"], [-1, 0, 0]);
    rotating(data["gt"], [-1, 0, 0]);
  }

  bound = [[1e10, -1e10], [1e10, -1e10], [1e10, -1e10]];
  fb(data["pred"]);
  fb(data["gt"]);

  config.tballsize = config.tballsize_base * (bound[0][1] - bound[0][0]) / 5; 
  config.ballsize = config.ballsize_base * (bound[0][1] - bound[0][0]) / 5; 

  let zoom = 0.5 * (bound[0][1] - bound[0][0]) / 3;
  camera.position.set(zoom * -6.098631031333716, zoom * 4.544654504527356, zoom * 14.427502035284304);
  camera.lookAt(0, 0, 0);

  config.dirlightRadius = 4 * (bound[0][1] - bound[0][0]) / 4;
  //data["analyzed"] = true;
}

function setupScene(scene) {
  let f = 1828.391959798995 * 2 / 1280; 
  //let fov = Math.atan2(1, f) * 2 * 180 / Math.PI;
  //console.log(fov);
  let fov = 28;
  camera = new THREE.PerspectiveCamera(fov, 1, 0.1, 1000 );
  let zoom = 0.5;
  camera.position.set(zoom * -6.098631031333716, zoom * 4.544654504527356, zoom * 14.427502035284304);
  camera.lookAt(0, 0, 0);

  views[0].camera = camera;
  for ( let ii = 1; ii < views.length; ++ ii ) {
    const view = views[ ii ];
    let c = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100 );
    c.position.fromArray( view.eye );
    c.up.fromArray( view.up );
    c.lookAt(view.lookAt[0], view.lookAt[1], view.lookAt[2]);
    view.camera = c;
  }
  const t = 5;
  dirLight1 = new THREE.DirectionalLight( 0xffffff, 0.8);
  dirLight1.position.set( 3, 8, 12);
  dirLight1.castShadow = true;
  dirLight1.shadow.radius = config.dirlightRadius;
  dirLight1.shadow.blurSamples = config.dirlightSamples;
  dirLight1.shadow.bias = -0.002;
  dirLight1.shadow.mapSize.width = 2048;
  dirLight1.shadow.mapSize.height = 2048;
  dirLight1.shadow.camera.left = -t;
  dirLight1.shadow.camera.right = t;
  dirLight1.shadow.camera.top = t;
  dirLight1.shadow.camera.bottom = -t;
  dirLight1.shadow.camera.near = 0.5; 
  dirLight1.shadow.camera.far = 50; 
  scene.add( dirLight1 );

  let light2 = new THREE.PointLight( 0xffffff, 0.3);
  light2.position.set(4, 8, 4);
  light2.castShadow = false;
  scene.add(light2);

  const ambientLight = new THREE.AmbientLight( 0xffffff, 0.4);
  scene.add( ambientLight );

  customAddTrajectory();

  addCheckerboard(config.patch_size, config.cb_size);
  drawFlatLine(scene, [-config.cb_size * 0.5, 0, config.cb_size * 0.5], [config.cb_size * 0.5, 0, config.cb_size * 0.5], 0.03, 1, 0xAAAAAA);

  drawFlatLine(scene, [-config.cb_size * 0.5, 0, config.cb_size * 0.5], [-config.cb_size * 0.5, 0, -config.cb_size * 0.5], 0.03, 1, 0xAAAAAA);
}

function genFakeData() {
  let px = -3, py = 0, pz = -2.5;
  let vx = 1, vy = 9, vz = 0.5;

  const tstep = 1 / 25.0;
  data[0]["pred"] = [];
  let dp = data[0]["pred"];
  for (let i = 0; i < 200; i++) {
    dp.push([px, py, pz]);
    px += vx * tstep;
    py += vy * tstep;
    pz += vz * tstep;
    if (py < 0) {
      py *= -1;
      vy *= -0.7;
    }
    vy -= 9.81 * tstep;
  }
  data[0]["gt"] = [];
  let rr = function(s) { return Math.random() * s * 2 - s;}; 
  let ra = function() { return [rr(0.1), rr(0.05), rr(0.1)];};

  let n = 15;
  let e1 = ra(), e2 = ra();
  for (let i = 0; i < dp.length; i++) {
    let t = (i % n) / n;

    data[0]["gt"].push([
      dp[i][0] + e2[0] * t + e1[0] * (1-t), 
      dp[i][1] + e2[1] * t + e1[1] * (1-t), 
      dp[i][2] + e2[2] * t + e1[2] * (1-t)]);

    if (i % n == n-1) {
      e1 = e2;
      e2 = ra();
    }

  }

}

readData(function () {
  $(document).ready(function() {
    if (window.location.hash) {
      config.traj_id = parseInt(window.location.hash.substring(1));
    }

    scene = new THREE.Scene();
    setupGUI();
    //genFakeData();

    //scene.background = new THREE.Color( 0xFFFFFF );
    //scene.fog = new THREE.FogExp2( 0xcccccc, 0.01 );

    renderer = new THREE.WebGLRenderer({antialias: true, preserveDrawingBuffer: true});
    document.getElementById("render").appendChild( renderer.domElement );
    //document.body.appendChild( renderer.domElement );
    renderer.shadowMap.enabled = config.shadow;
    renderer.shadowMap.type = THREE.VSMShadowMap;

    setupScene(scene);

    setSize();
    const controls = new THREE.OrbitControls( camera, renderer.domElement );
    controls.addEventListener( 'change', function() {
      //console.log("camera.position.set(" + camera.position.x + ", " + camera.position.y + ", " + camera.position.z + ");");
    });
    $(document).keydown(function(e) {
      if (e.which == 40) { // Down key
        config.traj_id ++;  
      } else if (e.which == 38) { // Up key
        config.traj_id --;  
      }
      config.traj_id = Math.min(Object.keys(data).length-1, Math.max(0, config.traj_id));

      if (e.which == 40 || e.which == 38) {
        console.log("Showing trajectory ", config.traj_id);
        customAddTrajectory();
      }
    });
    $("#savetopng").click(function(e) {
      e.preventDefault();
      var canvas = renderer.domElement;
      var dataURL = canvas.toDataURL("image/png");
      var newTab = window.open('about:blank','image from canvas');
      newTab.document.write("<img src='" + dataURL + "' alt='from canvas'/>");
    });
    
    $("#download").click(function(e) {
      e.preventDefault();
      var a = document.createElement('a');
      a.href = renderer.domElement.toDataURL().replace("image/png", "image/octet-stream");
      a.download = 'canvas.png';
      a.click();
    });

    // instantiate a loader
    const animate = function () {
      requestAnimationFrame( animate );

      const material = new THREE.MeshStandardMaterial( { color: 0xff0000} );

      anim += clock.getDelta() * config.speed;

      const FPS = 30.0;
      const duration = (traj.children.length - 1) / FPS;
      while (anim >= duration) anim -= duration;

      let id = anim * FPS;
      let id0 = Math.floor(id);
      let id1 = id0 + 1;

      ball.position.set(0, 0, 0);
      ball.position.addScaledVector(traj.children[id0].position, id1 - id);
      ball.position.addScaledVector(traj.children[id1].position, id - id0);

      if (config.drawtail > 0) {
        addTail2(id1, ball, id - id0);
      }

      render();
    };

    animate();
    //data["1"] = JSON.parse(JSON.stringify(data["0"]));
    //for (var i = 0; i < data["0"]["pred"].length; i++) {
      //data["1"]["pred"][i][2] *= -1;
      //}
    //document.body.innerHTML = JSON.stringify(data);
  });
});


function render() {
  let dim = [0, 0, 0];
  for (let i = 0; i < 3; i++) 
    dim[i] = (bound[i][1] - bound[i][0]) * 1.1;

  const smallw = views[1].width * vwidth;
  const smallh = views[1].height * vheight;

  let ratio = 0;
  ratio = Math.max(ratio, dim[0] / smallw);
  ratio = Math.max(ratio, dim[1] / smallh);
  ratio = Math.max(ratio, dim[2] / smallw);

  const simpleBlue = blueColoring(null, true);
  const simpleRed = redColoring(null, true);

  let oldBlue = traj.children[0].material;
  let oldRed = gt_traj.children[0].material;

  for ( let ii = 0; ii < views.length; ++ ii ) {
    const view = views[ ii ];
    const c = view.camera;

    const left = Math.floor( vwidth * view.left );
    const bottom = Math.floor( vheight * view.bottom );
    const width = Math.floor( vwidth * view.width );
    const height = Math.floor( vheight * view.height );
    c.aspect = width / height;

    renderer.setViewport( left, bottom, width, height );
    renderer.setScissor( left, bottom, width, height );
    renderer.setScissorTest( true );
    renderer.setClearColor( view.background );

    if (ii == 0) {
      renderer.shadowMap.enabled = config.shadow;
      scene.traverse(function (child) {
        if (child.material) {
          child.material.needsUpdate = true
        }
      })
      cb.visible = true;
    } else if (ii == 1 || ii == 2) { // XY plane
      cb.visible = false;

      c.top = ratio * smallh / 2;
      c.bottom = -c.top;
      c.right = ratio * smallw / 2;
      c.left = -c.right;
      if (ii == 1) {
        const px = (bound[0][0] + bound[0][1]) / 2;
        const py = bound[1][0] + (ratio * smallh) * 0.4; 
        c.position.set(px, py, config.cb_size+1);
        c.lookAt(px, py, 0);
      } else {
        const py = bound[1][0] + (ratio * smallh) * 0.4; 
        const pz = (bound[2][0] + bound[2][1]) / 2;
        c.position.set(-config.cb_size-1, py, -pz);
        c.lookAt(0, py, -pz);
      }
    } else {
      renderer.shadowMap.enabled = false;
      scene.traverse(function (child) {
        if (child.material) {
          child.material.needsUpdate = true
        }
      })
      cb.visible = true;
      let ratio2 = Math.max(dim[0] / smallw, dim[2] / smallh);

      c.top = ratio2 * smallh / 2;
      c.bottom = -c.top;
      c.right = ratio2 * smallw / 2;
      c.left = -c.right;

      const px = (bound[0][0] + bound[0][1]) / 2;
      const pz = (bound[2][0] + bound[2][1]) / 2;
      c.position.set(px, 10, -pz);
      c.lookAt(px, 0, -pz);
    }
    c.updateProjectionMatrix();

    if (ii == 1) {
      for (let i = 0; i < traj.children.length; i++) 
        traj.children[i].material = simpleBlue;
      for (let i = 0; i < gt_traj.children.length; i++) 
        gt_traj.children[i].material = simpleRed;
    } 
    renderer.render( scene, c );
  }
  for (let i = 0; i < traj.children.length; i++) 
    traj.children[i].material = oldBlue;
  for (let i = 0; i < gt_traj.children.length; i++) 
    gt_traj.children[i].material = oldRed;

  simpleBlue.dispose();
  simpleRed.dispose();
}
window.addEventListener( 'resize', setSize, false );

function setSize() {
  //console.log($("#render").width());
  //console.log($("#render").outerWidth());
  vwidth = $("#render").width();//window.innerWidth;// - $("#control").outerWidth();
  vheight = $("#render").height(); //800;//window.innerHeight;

  //camera.aspect = vwidth / vheight;
  //camera.updateProjectionMatrix();
  renderer.setSize(vwidth, vheight);
}

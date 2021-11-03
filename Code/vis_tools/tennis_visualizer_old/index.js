var anim = 0;
var clock = new THREE.Clock();

var camera;
var renderer;
var vwidth;
var vheight;

var scene;
var ball;
var tail;
var vr;

var traj, traj_line;

const config = {
  dirlightRadius: 5,
  dirlightSamples: 16,
  shadow: false,
  showall: false,
  showline: false,
  speed: 1,
  drawtail: 10,
  traj_id: 0,
  ballsize: 0.1,
  tballsize: 0.07
};

const views = [
  {
    left: 0,
    bottom: 0.2,
    width: 1.0,
    height: 0.8,
    background: new THREE.Color( 0x999999 ),
    eye: [ 0, 300, 1800 ],
    up: [ 0, 1, 0 ],
  },
  {
    left: 0,
    bottom: 0,
    width: 1/3.0,
    height: 0.2,
    background: new THREE.Color( 0.3, 0.3, 0.3),
    wview: w1 * 1.3, 
    eye: [ 0, 2, 20 ],
    lookAt: [0, 1.8, -20],
    up: [ 0, 1, 0 ],
  },
  {
    left: 1/3.0,
    bottom: 0,
    width: 1/3.0,
    height: 0.2,
    wview: h1 * 1.1,
    background: new THREE.Color( 0.3, 0.3, 0.3),
    eye: [ 20, 3, 0 ],
    lookAt: [-20, 2, 0],
    up: [ 0, 1, 0 ],
  },
  {
    left: 2/3.0,
    bottom: 0,
    width: 1/3.0,
    height: 0.2,
    wview: h1 * 1.1,
    background: new THREE.Color( 0.3, 0.3, 0.3 ),
    eye: [ 0, 40, 0],
    lookAt: [0, 0, 0],
    up: [ -1, 0, 0 ],
  }
];

var dirLight1;

function setupGUI() {
  const gui = new dat.GUI({width: 300});
  
  gui.useLocalStorage = true;

  const folder_traj = gui.addFolder( 'Trajectory' );
  folder_traj.add(config, 'traj_id', Object.keys(data)).name('Trajectory ID').listen().onChange(function(value) {
    console.log("Showing trajectory ", config.traj_id);
    addTrajectory(data[config.traj_id], ballColoring, lineColoring);
  });

  folder_traj.add(config, 'showall').name('Show Balls').listen().onChange( function(value) { 
    console.log("in", value);
    for (let i = 0; i < traj.children.length; i++) {
      traj.children[i].visible = value;
    }
    console.log(this);
  });
  folder_traj.add(config, 'showline').name('Show Line').listen().onChange( function(value) { 
    for (let i = 0; i < traj_line.children.length; i++) {
      traj_line.children[i].visible = value;
    }
  });
  folder_traj.add(config, 'drawtail', 0, 30, 1).name('Draw Tail').listen().onChange(function(value) {
    if (value == 0 && typeof tail !== "undefined") {
      scene.remove(tail);
      for (let i = 0; i < tail.children.length; i++) {
        tail.children[i].geometry.dispose();
        tail.children[i].material.dispose();
      }
    }
  });;

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
  dirlightFolder.add( config, 'dirlightRadius', 0, 25, 0.1).name( 'Shadow Radius' ).onChange( function ( value ) {
    dirLight1.shadow.radius = value;
  } );

  dirlightFolder.add( config, 'dirlightSamples', 1, 64, 1).name( 'Shadow Samples' ).onChange( function ( value ) {
    dirLight1.shadow.blurSamples = value;
  } );

  folder_traj.open();
  folder_render.open();
  //dirlightFolder.open();

}

function gradientColoring(point) {
  let depth = ((point[2] / (h1 * 1.3) + 1) * 0.5);

  if (depth < 0) depth = 0;
  if (depth > 1) depth = 1;

  let color1 = [255, 0, 0];
  let color2 = [255, 240, 0];

  return new THREE.Color(
    (color1[0] * depth + color2[0] * (1-depth)) / 255, 
    (color1[1] * depth + color2[1] * (1-depth)) / 255, 
    (color1[2] * depth + color2[2] * (1-depth)) / 255);
}
function ballColoring(point) {
  return new THREE.MeshStandardMaterial( { 
    color: gradientColoring(point), 
    opacity: 0.8,
    transparent: true} );
}
function lineColoring(point) {
  return new THREE.LineBasicMaterial( { color: gradientColoring(point)} );
}

function setupScene(scene) {
  let f = 1828.391959798995 * 2 / 1280; 
  let fov = Math.atan2(1, f) * 2 * 180 / Math.PI;
  camera = new THREE.PerspectiveCamera(fov, 1, 0.1, 1000 );
  camera.position.set(0, 8.6, 32.84);
  camera.lookAt(0, 0, 0);

  views[0].camera = camera;
  for ( let ii = 1; ii < views.length; ++ ii ) {
    const view = views[ ii ];
    let c = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 1000 );
    c.position.fromArray( view.eye );
    c.up.fromArray( view.up );
    c.lookAt(view.lookAt[0], view.lookAt[1], view.lookAt[2]);
    view.camera = c;
  }
  const t = 15;
  dirLight1 = new THREE.DirectionalLight( 0xffffff, 0.4);
  dirLight1.position.set( 10, 8, 12 );
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

  const ambientLight = new THREE.AmbientLight( 0xffffff, 0.85 );
  scene.add( ambientLight );

  drawCourt(scene);
  addTrajectory(data[config.traj_id], ballColoring, lineColoring);
}

readData(function () {
  $(document).ready(function() {
    //console.log(data[0]["pred"][0], data[1]["pred"][0]);
    scene = new THREE.Scene();
    setupGUI();
    //scene.background = new THREE.Color( 0x999999 );
    //scene.fog = new THREE.FogExp2( 0xcccccc, 0.01 );

    const queryString = window.location.search;
    const urlParams = new URLSearchParams(queryString);
    vr = (urlParams.get('vr') !== null);

    renderer = new THREE.WebGLRenderer({antialias: true});
    document.body.appendChild( renderer.domElement );
    renderer.shadowMap.enabled = false;
    renderer.shadowMap.type = THREE.VSMShadowMap;

    if (vr) {
      document.body.appendChild( VRButton.createButton( renderer ) );
      renderer.xr.enabled = true;
    }

    setupScene(scene);

    setSize();
    const controls = new THREE.OrbitControls( camera, renderer.domElement );
    $(document).keydown(function(e) {
      if (e.which == 40) { // Down key
        config.traj_id ++;  
      } else if (e.which == 38) { // Up key
        config.traj_id --;  
      }
      config.traj_id = Math.min(Object.keys(data).length-1, Math.max(0, config.traj_id));

      if (e.which == 40 || e.which == 38) {
        console.log("Showing trajectory ", config.traj_id);
        addTrajectory(data[config.traj_id], ballColoring, lineColoring);
      }
    });

    renderer.setAnimationLoop( function () {
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
      if (vr)
        renderer.render( scene, camera );
      else 
        render();

    } );
  });
});


function render() {
  for ( let ii = 0; ii < views.length; ++ ii ) {
    const view = views[ ii ];
    const c = view.camera;

    const left = Math.floor( vwidth * view.left );
    const bottom = Math.floor( vheight * view.bottom );
    const width = Math.floor( vwidth * view.width );
    const height = Math.floor( vheight * view.height );

    renderer.setViewport( left, bottom, width, height );
    renderer.setScissor( left, bottom, width, height );
    renderer.setScissorTest( true );
    renderer.setClearColor( view.background );

    if (ii == 0) {
      c.aspect = width / height;
    } else if (ii == 3) {
      c.left = -view.wview;
      c.right = -c.left;
      c.top = c.right * height / width;
      c.bottom = -c.top;
      if (c.top < w1 * 1.1) {
        c.top = w1 * 1.1;
        c.bottom = -c.top;
        c.right = c.top * width / height;
        c.left = -c.right;
      }
    } else {
      c.left = -view.wview;
      c.right = -c.left;
      c.top = c.right * height / width;
      c.bottom = -c.top;
    }
    c.updateProjectionMatrix();

    renderer.render( scene, c );
  }
}
window.addEventListener( 'resize', setSize, false );

function setSize() {
  console.log($("#control").width());
  vwidth = window.innerWidth;// - $("#control").outerWidth();
  vheight = window.innerHeight;

  camera.aspect = vwidth / vheight;
  camera.updateProjectionMatrix();
  renderer.setSize(vwidth, vheight);
}

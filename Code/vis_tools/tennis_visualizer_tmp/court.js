const w1 = 5.4868;
const h1 = 11.8872;

function drawFlatLine(scene, a, b, w, stand=0, color=0xffffff, castShadow=false) {
  if (Array.isArray(a)) 
    a = new THREE.Vector3(a[0], a[1], a[2]);
  if (Array.isArray(b)) 
    b = new THREE.Vector3(b[0], b[1], b[2]);

  if (stand)
    var up = new THREE.Vector3 (0, 0, 1);
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
  const material = new THREE.MeshStandardMaterial( { color: color, side: THREE.DoubleSide} );
  let geometry = new THREE.BufferGeometry()
  const points = [p[0], p[1], p[2], p[0], p[2], p[3]];
  geometry.setFromPoints(points);
  geometry.computeVertexNormals();

  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = castShadow;
  scene.add(mesh);
}

function drawCenteredRect(scene, w, h, y, color) {
  const material = new THREE.MeshStandardMaterial( { color: color, side: THREE.DoubleSide} );
  let geometry = new THREE.BufferGeometry()

  let p = [new THREE.Vector3 (-w/2, y, -h/2),
    new THREE.Vector3 (-w/2, y, h/2), 
    new THREE.Vector3 (w/2, y, h/2), 
    new THREE.Vector3 (w/2, y, -h/2)];
  const points = [p[0], p[1], p[2], p[0], p[2], p[3]];
  geometry.setFromPoints(points);
  geometry.computeVertexNormals();
  const mesh = new THREE.Mesh(geometry, material);
  mesh.receiveShadow = true;
  scene.add(mesh);
}


function drawCourt(scene) {

  let court = [[-w1, 0, -h1],
    [-w1, 0, h1],
    [w1, 0, h1], 
    [w1, 0, -h1], 
    [-4.1148, 0, -h1], 
    [-4.1148, 0, h1], 
    [4.1148, 0, h1],
    [4.1148, 0, -h1], 
    [-4.1148, 0, -6.4008],
    [4.1148, 0, -6.4008], 
    [-4.1148, 0, 6.4008],
    [4.1148, 0, 6.4008],
    [0, 0, -6.4008], 
    [0, 0, 6.4008],
    [-w1, 0, 0],
    [w1, 0, 0],
    ];
  let lines = [
    [0,1],[1,2],[2,3],[3,0],
    [4,5],[7,6],
    [8,9],[10,11],
    [12,13],
  ];
  const poleH = 1.0668;
  const poleHM = 0.9144;
  const polex = 0.9144+w1;

  for (let i = 0; i < court.length; i++) 
    court[i] = new THREE.Vector3(court[i][0], court[i][1], court[i][2]);
  for (let i = 0; i < lines.length; i++) 
    drawFlatLine(scene, court[lines[i][0]], court[lines[i][1]], 0.07); 
0.9144
  const pole = new THREE.CylinderGeometry( 0.05, 0.05, poleH, 16);
  const material = new THREE.MeshBasicMaterial( {color: 0x5a5a5a} );
  let cylinder = new THREE.Mesh( pole, material );
  cylinder.castShadow = true;
  cylinder.translateX(polex);
  cylinder.translateY(poleH/2);
  scene.add( cylinder );
  cylinder = cylinder.clone();
  cylinder.position.x *= -1;
  scene.add( cylinder );

  drawFlatLine(scene, [polex, poleH, 0], [w1, poleH, 0], 0.07, 1, 0xffffff, true); 
  drawFlatLine(scene, [-polex, poleH, 0], [-w1, poleH, 0], 0.07, 1, 0xffffff, true); 

  drawFlatLine(scene, [0, poleHM, 0], [w1, poleH, 0], 0.07, 1, 0xffffff, true); 
  drawFlatLine(scene, [0, poleHM, 0], [-w1, poleH, 0], 0.07, 1, 0xffffff, true); 

  drawFlatLine(scene, [0, poleHM, 0], [0, 0, 0], 0.05, 1, 0xffffff, true); 

  drawCenteredRect(scene, 30, 40, -0.011, 0x9ab389);
  drawCenteredRect(scene, w1*2, h1*2, -0.01, 0x706b8a);

  const netv = 20, neth = 60;
  for (let i = 0; i < netv; i++) 
    for (let j = -1; j <= 1; j+= 2) 
      drawFlatLine(scene, [polex * j, (poleH - 0.07) * i / (netv-1), 0], [0, (poleHM - 0.07) * i / (netv-1), 0], 0.005, 1, 0x000000, true); 

  for (let i = 0; i < neth; i++) 
    for (let j = -1; j <= 1; j+= 2) {
      drawFlatLine(scene, [j * polex * i / (neth - 1), 0, 0], [j * polex * i / (neth - 1), poleHM + (poleH - poleHM) * i / (neth-1) - 0.07, 0], 0.005, 1, 0x000000, true); 
    }

}

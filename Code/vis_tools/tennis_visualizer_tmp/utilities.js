function readData(done_callback) {
  const queryString = window.location.search;
  const urlParams = new URLSearchParams(queryString);
  const inputJSON = urlParams.get('input')
  console.log(inputJSON);

  if (inputJSON == null) {
    document.body.innerHTML = 'Please append "?input=DATA.json" to the url. <br>For example, <a href="index.html?input=tennis.json">index.html?input=tennis.json</a>'; 
  }

  var script = document.createElement('script');
  script.src = inputJSON;
  document.head.appendChild(script); //or something of the likes
  script.onload = done_callback;
}

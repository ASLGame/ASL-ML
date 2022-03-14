
const video3 = document.getElementsByClassName('input_video3')[0];
const out3 = document.getElementsByClassName('output3')[0];
const controlsElement3 = document.getElementsByClassName('control3')[0];
const canvasCtx3 = out3.getContext('2d');
const fpsControl = new FPS();

const spinner = document.querySelector('.loading');
spinner.ontransitionend = () => {
  spinner.style.display = 'none';
};



async function loadModel(){

  //loads model
  R_model = await tf.loadGraphModel('R_graph_model/model.json');
  L_model = await tf.loadGraphModel('L_graph_model/model.json');
  
  return {L_model, R_model};
}




let R_Model;
let L_Model;

loadModel().then(result => {
   R_Model = result.R_model;
   L_Model = result.L_model;
  
});





function calc_landmark_list(landmarks){
  width = 480;
  height = 480;

  let landmark_points = [];

  landmarks.forEach(value =>{
    landmark_x = Math.min(width-1, value.x * width);
    landmark_y = Math.min(height-1, value.y*height);
    landmark_points.push(landmark_x);
    landmark_points.push(landmark_y);
  });
  //console.log('Calc_landmark', landmark_points);
  return landmark_points;


}

function pre_process_landmarks(landmark_list){

  
  let base_x;
  let base_y;
  
  for( let i = 0; i < landmark_list.length; i += 2){
    if (i === 0){
      base_x = landmark_list[i];
      base_y = landmark_list[i+1];

    }

    landmark_list[i] = landmark_list[i] - base_x;
    landmark_list[i+1] = landmark_list[i+1] - base_y;
  }

  maxValue = Math.max.apply(null, landmark_list.map(Math.abs));

  //console.log('maxValue', maxValue);
  function normalize(n){
    return n/maxValue;
  }
  
  landmark_list = landmark_list.map(function(value){
    return normalize(value);
  });

  //console.log('pre_process', landmark_list);
  console.log('');
  return landmark_list;


}

function onResultsHands(results) {
  
  document.body.classList.add('loaded');
  fpsControl.tick();

 
  canvasCtx3.save();
  canvasCtx3.clearRect(0, 0, out3.width, out3.height);
  canvasCtx3.drawImage(
      results.image, 0, 0, out3.width, out3.height);
  if (results.multiHandLandmarks && results.multiHandedness) {
    for (let index = 0; index < results.multiHandLandmarks.length; index++) {
      const classification = results.multiHandedness[index];
      const isRightHand = classification.label === 'Right';
      const landmarks = results.multiHandLandmarks[index];
      //Preprocess landmarks
      let landmark_list = calc_landmark_list(landmarks);
      landmark_list = pre_process_landmarks(landmark_list);
      landmark_list = tf.tensor2d([landmark_list]);
      //console.log(landmark_list);
      let prediction;
      if(isRightHand){
        prediction = R_Model.predict(landmark_list);
      }else{
        prediction = L_Model.predict(landmark_list);
      }
      const scores = prediction.arraySync()[0];

      const maxScore = prediction.max().arraySync();
      const maxScoreIndex = scores.indexOf(maxScore);

      console.log(maxScoreIndex);

      

      drawConnectors(
          canvasCtx3, landmarks, HAND_CONNECTIONS,
          {color: isRightHand ? '#00FF00' : '#FF0000'}),
      drawLandmarks(canvasCtx3, landmarks, {
        color: isRightHand ? '#00FF00' : '#FF0000',
        fillColor: isRightHand ? '#FF0000' : '#00FF00',
        radius: (x) => {
          return lerp(x.from.z, -0.15, .1, 10, 1);
        }
      });
    }
  }
  canvasCtx3.restore();
}

const hands = new Hands({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.1/${file}`;
}});

hands.onResults(onResultsHands);

const camera = new Camera(video3, {
  onFrame: async () => {
    await hands.send({image: video3});
  },
  width: 480,
  height: 480
});
camera.start();

new ControlPanel(controlsElement3, {
      selfieMode: true,
      maxNumHands: 1,
      minDetectionConfidence: 0.75,
      minTrackingConfidence: 0.5
    })
    .add([
      new StaticText({title: 'MediaPipe Hands'}),
      fpsControl,
      new Toggle({title: 'Selfie Mode', field: 'selfieMode'}),
      new Slider(
          {title: 'Max Number of Hands', field: 'maxNumHands', range: [1, 4], step: 1}),
      new Slider({
        title: 'Min Detection Confidence',
        field: 'minDetectionConfidence',
        range: [0, 1],
        step: 0.01
      }),
      new Slider({
        title: 'Min Tracking Confidence',
        field: 'minTrackingConfidence',
        range: [0, 1],
        step: 0.01
      }),
    ])
    .on(options => {
      video3.classList.toggle('selfie', options.selfieMode);
      hands.setOptions(options);
    });


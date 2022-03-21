"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
/* eslint-disable prettier/prettier */
const myVideoElement = document.querySelector(".input_video2");
const myCanvasElement = document.querySelector('.output_canvas2');
const myCanvasCtx = myCanvasElement === null || myCanvasElement === void 0 ? void 0 : myCanvasElement.getContext('2d');
//--------------------------------------
function loadModel() {
    return __awaiter(this, void 0, void 0, function* () {
        // @ts-ignore
        const R_model = yield tf.loadGraphModel('R_graph_model/model.json');
        // @ts-ignore
        const L_model = yield tf.loadGraphModel('L_graph_model/model.json');
        return { L_model, R_model };
    });
}
let R_model;
let L_model;
loadModel().then(result => {
    R_model = result.R_model;
    L_model = result.L_model;
});
function calc_landmark_list(landmarks) {
    const width = 480;
    const height = 480;
    let landmark_points = [];
    landmarks.forEach((value) => {
        let landmark_x = Math.min(width - 1, value.x * width);
        let landmark_y = Math.min(height - 1, value.y * height);
        landmark_points.push(landmark_x);
        landmark_points.push(landmark_y);
    });
    //console.log('Calc_landmark', landmark_points);
    return landmark_points;
}
function pre_process_landmarks(landmark_list) {
    let base_x;
    let base_y;
    for (let i = 0; i < landmark_list.length; i += 2) {
        if (i === 0) {
            base_x = landmark_list[i];
            base_y = landmark_list[i + 1];
        }
        landmark_list[i] = landmark_list[i] - base_x;
        landmark_list[i + 1] = landmark_list[i + 1] - base_y;
    }
    const maxValue = Math.max.apply(null, landmark_list.map(Math.abs));
    //console.log('maxValue', maxValue);
    function normalize(n) {
        return n / maxValue;
    }
    landmark_list = landmark_list.map(function (value) {
        return normalize(value);
    });
    //console.log('pre_process', landmark_list);
    console.log('');
    return landmark_list;
}
//--------------------------------------
function NewonResults(result) {
    var _a, _b;
    const width = (_a = myCanvasElement === null || myCanvasElement === void 0 ? void 0 : myCanvasElement.width) !== null && _a !== void 0 ? _a : 0;
    const height = (_b = myCanvasElement === null || myCanvasElement === void 0 ? void 0 : myCanvasElement.height) !== null && _b !== void 0 ? _b : 0;
    myCanvasCtx === null || myCanvasCtx === void 0 ? void 0 : myCanvasCtx.save();
    myCanvasCtx === null || myCanvasCtx === void 0 ? void 0 : myCanvasCtx.clearRect(0, 0, width, height);
    myCanvasCtx === null || myCanvasCtx === void 0 ? void 0 : myCanvasCtx.drawImage(result.image, 0, 0, width, height);
    //console.log(result);
    if (result.multiHandLandmarks && result.multiHandedness) {
        for (let index = 0; index < result.multiHandLandmarks.length; index++) {
            const classification = result.multiHandedness[index];
            const isRightHand = classification.label === 'Right';
            const landmarks = result.multiHandLandmarks[index];
            //Preprocess Landmarks
            let landmark_list = calc_landmark_list(landmarks);
            landmark_list = pre_process_landmarks(landmark_list);
            //@ts-ignore
            landmark_list = tf.tensor2d([landmark_list]);
            let prediction;
            if (isRightHand) {
                prediction = R_model === null || R_model === void 0 ? void 0 : R_model.predict(landmark_list);
            }
            else {
                prediction = L_model === null || L_model === void 0 ? void 0 : L_model.predict(landmark_list);
            }
            const scores = prediction.arraySync()[0];
            const maxScore = prediction.max().arraySync();
            const maxScoreIndex = scores.indexOf(maxScore);
            console.log(maxScoreIndex);
            // eslint-disable-next-line
            // @ts-ignore
            drawConnectors(myCanvasCtx, landmarks, HAND_CONNECTIONS, {
                color: "#00CC00",
                lineWidth: 5,
            });
            // eslint-disable-next-line
            // @ts-ignore
            drawLandmarks(myCanvasCtx, landmarks, {
                color: "#FF0000",
                lineWidth: 2,
            });
        }
    }
    myCanvasCtx === null || myCanvasCtx === void 0 ? void 0 : myCanvasCtx.restore();
}
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
const hands = new Hands({ locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.1/${file}`;
    } });
hands === null || hands === void 0 ? void 0 : hands.setOptions({
    selfieMode: true,
    maxNumHands: 1,
    minDetectionConfidence: 0.75,
    minTrackingConfidence: 0.5
});
hands.onResults(NewonResults);
// eslint-disable-next-line
// @ts-ignore
const camera = new Camera(myVideoElement, {
    onFrame: () => __awaiter(void 0, void 0, void 0, function* () {
        yield hands.send({ image: myVideoElement });
    }),
    width: 1280,
    height: 720,
});
camera.start();

import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;
import kNear from "./knear.js"
const k = 2;
const machine = new kNear(k);
console.log(machine);
let dataset = [];
await fetch('./facesign_data.json')
    .then((response) => response.json())
    .then((json) => dataset = json);

for (let i = 0; i < dataset.length; i++) {
    machine.learn(dataset[i].pose, dataset[i].label);
}

dataset.sort(() => Math.random() - 0.5);

const trainSize = Math.floor(dataset.length * 0.8);
const trainData = dataset.slice(0, trainSize);
const testData = dataset.slice(trainSize);

for (let i = 0; i < trainData.length; i++) {
    machine.learn(trainData[i].pose, trainData[i].label);
}
console.log(testData)

let correctPredictions = 0;
let totalTestPoses = 0;

for (const testPose of testData) {
    const prediction = machine.classify(testPose.pose);
    console.log(`Predicted: ${prediction}, Actual: ${testPose.label}`);

    if (prediction === testPose.label) {
        correctPredictions++;
    }
    totalTestPoses++;
    console.log(totalTestPoses)
}

const accuracy = (correctPredictions / totalTestPoses) * 100;

console.log(`Accuracy: ${accuracy.toFixed(2)}%`);




//
// for (let i = 0; i < dataSet.length; i++){
//     machine.learn(dataSet[i].pose, dataSet[i].label)
// }









const demosSection = document.getElementById("demos");
const imageBlendShapes = document.getElementById("image-blend-shapes");
const videoBlendShapes = document.getElementById("video-blend-shapes");
let faceLandmarker;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;
const videoWidth = 480;
let selectedFaceSign = "happy";
let faceSignData = [];

async function createFaceLandmarker() {
    const filesetResolver = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm");
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
            delegate: "GPU"
        },
        outputFaceBlendshapes: true,
        runningMode,
        numFaces: 1
    });
    demosSection.classList.remove("invisible");
}
createFaceLandmarker();


const flattenCoordinates = (landmarks) => {
    const flattenedCoordinates = [];
    landmarks.forEach((landmark) => {
        flattenedCoordinates.push(landmark.x, landmark.y, landmark.z);
    });
    return flattenedCoordinates;
};


const imageContainers = document.getElementsByClassName("detectOnClick");
for (let imageContainer of imageContainers) {
    imageContainer.children[0].addEventListener("click", faceleClick);
}
async function faceleClick(event) {
    if (!faceLandmarker) {
        console.log("Wait for faceLandmarker to load before clicking!");
        return;
    }
    if (runningMode === "VIDEO") {
        runningMode = "IMAGE";
        await faceLandmarker.setOptions({ runningMode });
    }
    // Remove all landmarks drawed before
    const allCanvas = event.target.parentNode.getElementsByClassName("canvas");
    for (var i = allCanvas.length - 1; i >= 0; i--) {
        const n = allCanvas[i];
        n.parentNode.removeChild(n);
    }
    let faceLandmarks = faceLandmarker.detect(event.target);

    // Transformeer de data naar een enkele array van getallen
    let landmarksData = faceLandmarks.map(landmark => [landmark.x, landmark.y, landmark.z]).reduce((a, b) => a.concat(b), []);

    // Voeg een label toe aan de data
    let labeledData = { pose: landmarksData, label: "Blij" };

    // Sla de data op in een array
    let data = [];
    data.push(labeledData);

    // Sla de data op in een JSON-bestand
    const fs = require('fs');
    fs.writeFileSync('landmarksData.json', JSON.stringify(data));




    const faceLandmarkerResult = faceLandmarker.detect(event.target);
    console.log(faceLandmarkerResult.faceLandmarks);
    const canvas = document.createElement("canvas");
    canvas.setAttribute("class", "canvas");
    canvas.setAttribute("width", event.target.naturalWidth + "px");
    canvas.setAttribute("height", event.target.naturalHeight + "px");
    canvas.style.left = "0px";
    canvas.style.top = "0px";
    canvas.style.width = `${event.target.width}px`;
    canvas.style.height = `${event.target.height}px`;
    event.target.parentNode.appendChild(canvas);
    const ctx = canvas.getContext("2d");
    const drawingUtils = new DrawingUtils(ctx);
    for (const landmarks of faceLandmarkerResult.faceLandmarks) {
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "#C0C0C070", lineWidth: 1 });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: "#FF3030" });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: "#FF3030" });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: "#30FF30" });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, { color: "#30FF30" });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: "#E0E0E0" });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, {
            color: "#E0E0E0"
        });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, { color: "#FF3030" });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, { color: "#30FF30" });
    }
    drawBlendShapes(imageBlendShapes, faceLandmarkerResult.faceBlendshapes);
}

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
// Check if webcam access is supported.
function hasGetUserMedia() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}
// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);
}
else {
    console.warn("getUserMedia() is not supported by your browser");
}
// Enable the live webcam view and start detection.
function enableCam(event) {
    if (!faceLandmarker) {
        console.log("Wait! faceLandmarker not loaded yet.");
        return;
    }
    if (webcamRunning === true) {
        webcamRunning = false;
        enableWebcamButton.innerText = "ENABLE PREDICTIONS";
    }
    else {
        webcamRunning = true;
        enableWebcamButton.innerText = "DISABLE PREDICTIONS";
    }
    // getUsermedia parameters.
    const constraints = {
        video: true
    };
    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    });
}
let lastVideoTime = -1;
let results = undefined;
const drawingUtils = new DrawingUtils(canvasCtx);
async function predictWebcam() {
    const radio = video.videoHeight / video.videoWidth;
    video.style.width = videoWidth + "px";
    video.style.height = videoWidth * radio + "px";
    canvasElement.style.width = videoWidth + "px";
    canvasElement.style.height = videoWidth * radio + "px";
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;

    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await faceLandmarker.setOptions({ runningMode: runningMode });
    }

    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        results = faceLandmarker.detectForVideo(video, startTimeMs);
    }

    if (results.faceLandmarks) {
        for (const landmarks of results.faceLandmarks) {
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "#C0C0C070", lineWidth: 1 });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: "#FF3030" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: "#FF3030" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: "#30FF30" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, { color: "#30FF30" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: "#E0E0E0" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: "#E0E0E0" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, { color: "#FF3030" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, { color: "#30FF30" });

            let currentEmotion = machine.classify(flattenCoordinates(landmarks));

            if (currentEmotion === 'happy') {
                paddleX -= 5;
            } else if (currentEmotion === 'sad') {
                paddleX += 5;
            } else if (currentEmotion === 'normal') {
                // Do nothing
            }

            if (paddleX < 0) {
                paddleX = 0;
            } else if (paddleX > canvas.width - PADDLE_WIDTH) {
                paddleX = canvas.width - PADDLE_WIDTH;
            }
        }
    }

    drawBlendShapes(videoBlendShapes, results.faceBlendshapes);

    if (webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
    }
}








function drawBlendShapes(el, blendShapes) {
    if (!blendShapes.length) {
        return;
    }
    // console.log(blendShapes[0]);
    let htmlMaker = "";
    blendShapes[0].categories.map((shape) => {
        htmlMaker += `
<!--      <li class="blend-shapes-item">-->
     <!--    <span class="blend-shapes-label">${shape.displayName || shape.categoryName}</span> -->
        <!--  <span class="blend-shapes-value" style="width: calc(${+shape.score * 100}% - 120px)">${(+shape.score).toFixed(4)}</span> -->
<!--      </li>-->
    `;
    });
    el.innerHTML = htmlMaker;
}






let canvas, canvasContext;
let emotionState = "normal";
// Bricks
const BRICK_W = 80;
const BRICK_H = 20;
const BRICK_GAP = 2;
const BRICK_COLS = 10;
const BRICK_ROWS = 14;
const brickGrid = new Array(BRICK_COLS*BRICK_ROWS);
let brickCount = 0;

// Ball
let ballX = 75;
let ballSpeedX = 8;
let ballY = 75;
let ballSpeedY = 8;

// Main Paddle
let paddleX = 400;
const PADDLE_THICKNESS = 15;
const PADDLE_WIDTH = 100;
const PADDLE_DIST_FROM_EDGE = 60;

// Mouse
let mouseX = 0;
let mouseY = 0;

/**********
 General GamePlay
 ***********/
window.onload = function(){
    canvas = document.getElementById('gameCanvas');
    canvasContext = canvas.getContext('2d');
    const framesPerSecond = 30;
    setInterval(updateAll, 1000/framesPerSecond);

    canvas.addEventListener('mousemove', updateMousePos);
    brickReset();
    ballRest();
}

function updateAll(){
    movement();
    playArea();
}

function ballRest(){
    ballX = canvas.width/2;
    ballY = canvas.height/2;
}

function brickReset(){
    brickCount = 0;
    let i;
    for (i = 0; i < 3 * BRICK_COLS; i++) {
        brickGrid[i] = false;
    }
    for (; i<BRICK_COLS*BRICK_ROWS; i++) {
        if(Math.random()<0.5){
            brickGrid[i] = true;
        } else {
            brickGrid[i] = false;
        }
        brickGrid[i] = true;
        brickCount++;
    }
}

function ballMove(){
    // ballMovement
    ballX += ballSpeedX;
    ballY += ballSpeedY;
    // ballY
    if(ballY > canvas.height){
        // ballSpeedY = -ballSpeedY;
        ballRest();
        brickReset();
    } else if(ballY < 0 && ballSpeedY > 0.0){
        ballSpeedY = -ballSpeedY;
    }
    // ballx
    if(ballX > canvas.width && ballSpeedX > 0.0){
        ballSpeedX = -ballSpeedX;
    } else if(ballX < 0 && ballSpeedX < 0.0){
        ballSpeedX = -ballSpeedX;
    }
}

function isBrickAtColRow(col, row){
    if (col >= 0 && col < BRICK_COLS &&
        row >= 0 && row < BRICK_ROWS) {
        const brickIndexUnderCoord= rowColToArrayIndex(col, row);
        return brickGrid[brickIndexUnderCoord];
    } else{
        return false;
    }
}

function ballBrickColl(){
    const ballBrickCol = Math.floor(ballX / BRICK_W);
    const ballBrickRow = Math.floor(ballY / BRICK_H);
    const brickIndexUnderBall = rowColToArrayIndex(ballBrickCol, ballBrickRow);
    if (ballBrickCol >= 0 && ballBrickCol < BRICK_COLS && ballBrickRow >= 0 && ballBrickRow < BRICK_ROWS){
        if (isBrickAtColRow(ballBrickCol, ballBrickRow)) {
            brickGrid[brickIndexUnderBall] = false;
            brickCount--;

            const prevBallX = ballX - ballSpeedX;
            const prevBallY = ballY - ballSpeedY;
            const prevBrickCol = Math.floor(prevBallX / BRICK_W);
            const prevBrickRow = Math.floor(prevBallY / BRICK_H);


            let bothTestFailed = true;

            if(prevBrickCol != ballBrickCol){
                if(isBrickAtColRow(prevBrickCol, ballBrickRow) == false){
                    ballSpeedX = -ballSpeedX;
                    bothTestFailed = false;
                }
            }

            if(prevBrickRow != ballBrickRow){
                if (isBrickAtColRow(ballBrickCol, prevBrickRow) == false) {
                    ballSpeedY = -ballSpeedY;
                    bothTestFailed = false;
                }
            }

            if(bothTestFailed){
                ballSpeedX = -ballSpeedX;
                ballSpeedY = -ballSpeedY;
            }

        }
    }
    // colorText(ballBrickCol+","+ballBrickRow+": "+brickIndexUnderBall, mouseX, mouseY, 'white');
}

function paddleMove(){
    // paddle
    const paddleTopEdgeY = canvas.height-PADDLE_DIST_FROM_EDGE;
    const paddleBottomEdgeY = paddleTopEdgeY+PADDLE_THICKNESS;
    const paddleLeftEdgeX = paddleX;
    const paddleRightEdgeX = paddleX+PADDLE_WIDTH;
    if(ballY > paddleTopEdgeY && // top of paddle
        ballY < paddleBottomEdgeY && // bottom of paddle
        ballX > paddleLeftEdgeX && // left half of paddle
        ballX < paddleRightEdgeX // right half of paddle
    ){

        ballSpeedY = -ballSpeedY;

        const paddleCenterX = paddleX + PADDLE_WIDTH/2;
        const ballDistFromCenterX = ballX - paddleCenterX;
        ballSpeedX = ballDistFromCenterX * 0.35;

        if (brickCount == 0) {
            brickReset();
        }

    }
}

function movement(){
    ballMove();
    ballBrickColl();
    paddleMove();
}

// function updateMousePos(evt) {
//     const rect = canvas.getBoundingClientRect();
//     const root = document.documentElement;
//
//     mouseX = evt.clientX - rect.left - root.scrollLeft;
//     mouseY = evt.clientY - rect.top - root.scrollTop;
//
//     paddleX = mouseX - PADDLE_WIDTH/2;
//
//     //cheat to test ball in any position
//     // ballX = mouseX;
//     // ballY = mouseY;
//     // ballSpeedY = 4;
//     // ballSpeedY = -4;
// }

/**********
 GamePlay Draw functions
 ***********/
function playArea(){
    // gameCanvas
    colorRect(0,0,canvas.width, canvas.height, 'white');
    // ball
    colorCircle();
    // paddle
    colorRect(paddleX, canvas.height-PADDLE_DIST_FROM_EDGE, PADDLE_WIDTH, PADDLE_THICKNESS, 'black');

    drawbricks();
}

function colorRect(leftX, topY, width, height, color){
    canvasContext.fillStyle = color;
    canvasContext.fillRect(leftX, topY, width, height);
}

function colorText(showWords, textX,textY, fillColor) {
    canvasContext.fillStyle = fillColor;
    canvasContext.fillText(showWords, textX, textY);
}

function rowColToArrayIndex(col, row){
    return col + BRICK_COLS * row;
}

function drawbricks(){
    for (let eachRow=0; eachRow<BRICK_ROWS; eachRow++) {
        for(let eachCol=0; eachCol<BRICK_COLS; eachCol++){
            const arrayIndex = rowColToArrayIndex(eachCol, eachRow);
            if(brickGrid[arrayIndex]){
                colorRect(BRICK_W*eachCol , BRICK_H*eachRow,
                    BRICK_W-BRICK_GAP, BRICK_H-BRICK_GAP, 'green');
            } //   if brick
        }// each brick
    }// each brickrow
}// drawbricks

function colorCircle(){
    canvasContext.fillStyle = 'black';
    canvasContext.beginPath();
    canvasContext.arc(ballX, ballY, 10, 0, Math.PI*2, true);
    canvasContext.fill();
}














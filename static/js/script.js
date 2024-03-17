// script.js

// This is a stub for any future navbar-related JavaScript you may want to add
document.addEventListener("DOMContentLoaded", () => {
  // This function will run after the DOM is fully loaded
  // Any navbar interactivity can go here

  // Example: Toggle mobile menu
  const menuButton = document.querySelector(".menu-button"); // Your menu button
  const navbarLinks = document.querySelector(".navbar-links"); // Your navbar links

  if (menuButton && navbarLinks) {
    menuButton.addEventListener("click", () => {
      // Toggle the .active class on the navbarLinks
      navbarLinks.classList.toggle("active");
    });
  }
});

/********************************************************* */
let sessionId = '';
const cameraButton = document.getElementById("cameraButton");
const stopCameraButton = document.getElementById("stopCameraButton");
// const downloadAttendanceButton = document.getElementById('downloadAttendanceButton');
const cameraStream = document.getElementById("cameraStream");
const modelSelect = document.getElementById("modelSelect");



const mySwitch = document.getElementById('generateReportSwitch');

const main_container = document.getElementById("main_container");
const placeholder = document.getElementById("placeholder-container");
const switch_container = document.getElementById("switch_ui");

//   const modelSetup = document.getElementById("modelSetup");

const processedStream = document.getElementById("processedStream");
let selectedModel = "";
let stream;
var CNN_predictions = [];
var Chehra_predictions = [];
var Yollo_predictions = [];


let startTime='';
let endTime='';

let averages = [0,0,0,0,0,0,0,0]; 
let frameCount=0;

var emotionChart;
var values = [0, 0, 0, 0, 0, 0, 0, 0];

let isSwitchOn = false;


function handleModelChange() {
  const selectedValue = modelSelect.value;
  if (cameraStream.srcObject != "") {
    stopCamera(false);
  }
  // Stop processing frames for the previous model

  // Clear previous stream and canvas
  processedStream.src = "";

  // Add your logic here based on the selected radio button
  if (selectedValue === "CNN") {
    console.log("CNN");
    selectedModel = "CNN";
    labels=["Anger",'Contempt', 'Disgust',"Fear","Happiness","Neutrality","Sadness","Surprise"];
    createChart(labels);
    openCamera();
  } else if (selectedValue === "Chehra") {
    console.log("Chehra");
    selectedModel = "Chehra";
    labels = ["Anger","Fear","Happiness","Neutrality","Sadness","Surprise"];
    createChart(labels);
    openCamera();
  } else if (selectedValue === "Yollo") {
    selectedModel = "Yollo";
    labels=["Angry", "Disgusted", "Fearful","Happy", "Neutral","Sad","Surprised"];
    createChart(labels);
    openCamera();
  } else {
    //show toast
    showToast("Please Select Any Model.");
  }
}

function openCamera() {
  cameraButton.style.display = "none";
  placeholder.style.display="none";
  // modelSelect.style.display="none";
  // modelSetup.style.display="none";

  stopCameraButton.style.display = "block";
  cameraStream.style.display = "block";
  processedStream.style.display = "block";
  main_container.style.display = "grid";
  switch_container.style.display="none";

   startTime = getCurrentTimestamp();

  isSwitchOn = mySwitch.checked;
  if(isSwitchOn){
    sessionId = generateSessionId();
    document.getElementById('sessionID').innerHTML=sessionId;
  }

  

  navigator.mediaDevices
    .getUserMedia({ video: true })
    .then((mediaStream) => {
      stream = mediaStream;
      cameraStream.srcObject = stream;
    })
    .catch((error) => console.error("Error accessing the camera:", error));

  // Start sending and displaying processed frames
  if (selectedModel === "CNN") {
    startSendingCNNFrames();
  } else if (selectedModel === "Chehra") {
    startSendingChehraFrames();
  } else if (selectedModel === "Yollo") {
    startSendingYolloFrames();
  }
  // startSendingFrames();
}


const createFolderAndFile = async (sessionId, modelName, content) => {
  try {
    const response = await fetch(`/create_file/${sessionId}/${modelName}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ content })  // Send content in JSON format
    });

    if (!response.ok) {
      throw new Error("Network response was not ok");
    }

    const result = await response.text();
    console.log(result);
  } catch (error) {
    console.error('Error:', error);
  }
};



function createSessionSummary(sessionId, startTime, endTime, selectedModel, labels, averages) {
  // Calculate session time
  const sessionStartTime = new Date(startTime);
  const sessionEndTime = new Date(endTime);
  const sessionDuration = (sessionEndTime - sessionStartTime) / 1000; // Duration in seconds

  // Start building the summary string
  let summary = `Session ID: ${sessionId}\n`;
  summary += `Start Time: ${sessionStartTime.toISOString()}\n`;
  summary += `End Time: ${sessionEndTime.toISOString()}\n`;
  summary += `Session Time: ${sessionDuration} seconds\n`;
  summary += `Selected Model: ${selectedModel}\n\n`;

  summary += `Average Emotions(%):\n`;
  // Append each label with its corresponding average value
  labels.forEach((label, index) => {
      summary += `${label}: ${averages[index]}\n`;
  });

  return summary;
}


function getCurrentTimestamp() {
  return new Date().toISOString();
}

function stopCamera(flag) {

  if (stream) {
    stream.getTracks().forEach((track) => track.stop());


    if(isSwitchOn){
        // create file start
        endTime = getCurrentTimestamp();
        for (var i = 0; i < labels.length; i++) {
          // averages[i] = averages[i]/frameCount;
          averages[i] = parseFloat((averages[i] / frameCount).toFixed(2));
        }

        const sessionSummary = createSessionSummary(sessionId, startTime, endTime, selectedModel, labels, averages);
        createFolderAndFile(sessionId,selectedModel, sessionSummary);

        startTime='';
        endTime='';
        averages = [0,0,0,0,0,0,0,0]; 
        frameCount=0;
        // create file end
    }

    if(flag){
      modelSelect.value="";
      selectedModel="";
      mySwitch.checked = false;
      isSwitchOn = false;
    }
  }
  stream = null;

  stopCameraButton.style.display = "none";
  main_container.style.display = "none";

  cameraButton.style.display = "block";
  placeholder.style.display="block";
  switch_container.style.display="flex";

  // modelSelect.style.display="block";
  // modelSetup.style.display="flex !important";

  cameraStream.style.display = "none";
  processedStream.style.display = "none";

  cameraStream.srcObject = null;
  processedStream.src = ""; // Clear the processed image
}

const get_chehra_predictions = async () => {
  try {
    const response = await fetch("/get_chehra_predictions");

    if (!response.ok) {
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    /** here you can call the chart function and pass the data.predictions 
          as a parameter to get updated chart everytime you run, till you stopCamera */
    //for chehra the lables are:  labels = ["anger","fear","happiness","neutrality","sadness","surprise"]
    if (data.predictions.length > 0) {
      Chehra_predictions = data.predictions;
      //update Chart
      updateChart(Chehra_predictions);
    }
  } catch (error) {
    console.error("Error fetching predictions:", error);
  }
};

function startSendingChehraFrames() {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = cameraStream.videoWidth;
  canvas.height = cameraStream.videoHeight;

  ctx.drawImage(cameraStream, 0, 0, canvas.width, canvas.height);
  get_chehra_predictions();
  canvas.toBlob(
    (blob) => {
      const formData = new FormData();
      formData.append("frame", blob);

      fetch("/process_chehra_frame", {
        method: "POST",
        body: formData,
      })
        .then((response) => response.blob())
        .then((processedBlob) => {
          const processedUrl = URL.createObjectURL(processedBlob);
          processedStream.src = processedUrl;

          // Recursively send and display processed frames
          if (selectedModel === "Chehra") {
            startSendingChehraFrames();
          }
        });
    },
    "image/jpeg",
    0.9
  );
}

const get_CNN_predictions = async () => {
  try {
    const response = await fetch("/get_cnn_predictions");

    if (!response.ok) {
      throw new Error("Network response was not ok");
    }

    /** here you can call the chart function and pass the data.predictions 
          as a parameter to get updated chart everytime you run, till you stopCamera */
    //for CNN the lables are: lables = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutrality', 'sadness', 'surprise']
    const data = await response.json();
    if (data.predictions.length > 0) {
      CNN_predictions = data.predictions;
      //update Chart
      updateChart(CNN_predictions);

    }
  } catch (error) {
    console.error("Error fetching predictions:", error);
  }
};

function startSendingCNNFrames() {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = cameraStream.videoWidth;
  canvas.height = cameraStream.videoHeight;

  ctx.drawImage(cameraStream, 0, 0, canvas.width, canvas.height);
  get_CNN_predictions();
  canvas.toBlob(
    (blob) => {
      const formData = new FormData();
      formData.append("frame", blob);

      fetch("/process_CNN_frame", {
        method: "POST",
        body: formData,
      })
        .then((response) => response.blob())
        .then((processedBlob) => {
          const processedUrl = URL.createObjectURL(processedBlob);
          processedStream.src = processedUrl;

          // Recursively send and display processed frames
          if (selectedModel === "CNN") {
            startSendingCNNFrames();
          }
        });
    },
    "image/jpeg",
    0.9
  );
}

const get_Yollo_predictions = async () => {
  try {
    const response = await fetch("/get_yollo_predictions");

    if (!response.ok) {
      throw new Error("Network response was not ok");
    }

    /** here you can call the chart function and pass the data.predictions 
          as a parameter to get updated chart everytime you run, till you stopCamera */
    //for CNN the lables are: lables  = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]
    const data = await response.json();
    if (data.predictions.length > 0) {
      Yollo_predictions = data.predictions;
      //update Chart
      updateChart(Yollo_predictions);

    }
  } catch (error) {
    console.error("Error fetching predictions:", error);
  }
};

function startSendingYolloFrames() {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = cameraStream.videoWidth;
  canvas.height = cameraStream.videoHeight;

  ctx.drawImage(cameraStream, 0, 0, canvas.width, canvas.height);
  get_Yollo_predictions();
  canvas.toBlob(
    (blob) => {
      const formData = new FormData();
      formData.append("frame", blob);

      fetch("/process_yollo_frame", {
        method: "POST",
        body: formData,
      })
        .then((response) => response.blob())
        .then((processedBlob) => {
          const processedUrl = URL.createObjectURL(processedBlob);
          processedStream.src = processedUrl;

          // Recursively send and display processed frames
          if (selectedModel === "Yollo") {
            startSendingYolloFrames();
          }
        });
    },
    "image/jpeg",
    0.9
  );
}

// function downloadAttendance() {
//     fetch('/download_attendance')
//         .then(response => response.blob())
//         .then(blob => {
//             const url = window.URL.createObjectURL(blob);
//             const a = document.createElement('a');
//             a.href = url;
//             a.download = 'attendance.csv';
//             document.body.appendChild(a);
//             a.click();
//             document.body.removeChild(a);
//             window.URL.revokeObjectURL(url);
//         })
//         .catch(error => console.error('Error downloading attendance:', error));
// }

/************************************************************ */
/********************************************************* */

function createChart(labels){
  var ctx = document.getElementById("emotionChart").getContext("2d");
  if (emotionChart) {
    emotionChart.destroy();
}

emotionChart = new Chart(ctx, {
  type: "bar",
  data: {
    labels: labels,
    datasets: [
      {
        label: "Emotion Analysis",
        data: values,
        backgroundColor: [
          "rgba(255, 0, 0, 0.5)",
          "rgba(128, 128, 128, 0.5)",
          "rgba(0, 128, 0, 0.5)",
          "rgba(0, 0, 255, 0.5)",
          "rgba(255, 255, 0, 0.5)",
          "rgba(192, 192, 192, 0.5)",
          "rgba(128, 0, 128, 0.5)",
          "rgba(255, 165, 0, 0.5)",
        ],
        borderColor: [
          "rgba(255, 0, 0, 1)",
          "rgba(128, 128, 128, 1)",
          "rgba(0, 128, 0, 1)",
          "rgba(0, 0, 255, 1)",
          "rgba(255, 255, 0, 1)",
          "rgba(192, 192, 192, 1)",
          "rgba(128, 0, 128, 1)",
          "rgba(255, 165, 0, 1)",
        ],
        borderWidth: 1,
      },
    ],
  },
  options: {
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
      },
    },
  },
});
}

function updateChart(predictions) {
  console.log("predictions: ", predictions);

  // Clear existing list items
  var emotionList = document.getElementById("emotion_info").querySelector("ul");
  emotionList.innerHTML = '';

  console.log("len::" + labels.length);
  frameCount += 1; // Increment frame count

  // Check if predictions structure is consistent
  let processedPredictions = (selectedModel === "Yollo") ? predictions : predictions[0];

  let processedValues = processedPredictions.map(value => (value * 100).toFixed(2));

    // Update the chart data synchronously
    emotionChart.data.datasets[0].data = processedValues.map(value => parseFloat(value));
    console.log("Processed values for chart update:", processedValues);

    // Use requestAnimationFrame for synchronized UI updates
    requestAnimationFrame(() => {
        // Refresh the chart to reflect new data
        emotionChart.update();

        // Clear existing list items
        var emotionList = document.getElementById("emotion_info").querySelector("ul");
        emotionList.innerHTML = '';

        // Synchronously update list items with the same data used for the chart
        processedValues.forEach((value, i) => {
            let floatValue = parseFloat(value);
            averages[i]=averages[i]+floatValue;
            
            let listItem = document.createElement("li");
            listItem.textContent = `${labels[i]}: ${value}%`;
            emotionList.appendChild(listItem);
        });
    });
}



// function updateChart(predictions) {
//   console.log("predictions: ", predictions);
//   // debugger;

//    // Clear existing list items
//    var emotionList = document.getElementById("emotion_info").querySelector("ul");
//    emotionList.innerHTML = '';

//   console.log("len::"+labels.length);
//   frameCount = frameCount+1;
//   for (var i = 0; i < labels.length; i++) {
//     if(selectedModel==="Yollo"){
//       if(predictions[i]!=null){
//         var value = predictions[i] * 100;
//         values[i] = value;
//         console.log(labels[i] + " " + values[i])
//         averages[i]=averages[i]+values[i];
  
        //  // Create list item for each emotion
        //  var listItem = document.createElement("li");
        //  listItem.textContent = labels[i] + ": " + value.toFixed(2) + "%";
        //  emotionList.appendChild(listItem);

        //  // Create list item for each emotion
        //  var listItem = document.createElement("li");
        //  listItem.textContent = labels[i] + ": " + value.toFixed(2) + "%";
        //  emotionList.appendChild(listItem);
  
//          // Create list item for each emotion
//          var listItem = document.createElement("li");
//          listItem.textContent = labels[i] + ": " + value.toFixed(2) + "%";
//          emotionList.appendChild(listItem);
  
//       }
//     }else{
//       if(predictions[0][i]!=null){
//         var value = predictions[0][i] * 100;
//         values[i] = value;
//         console.log(labels[i] + " " + values[i])
//         averages[i]=averages[i]+values[i];

//          // Create list item for each emotion
//          var listItem = document.createElement("li");
//          listItem.textContent = labels[i] + ": " + value.toFixed(2) + "%";
//          emotionList.appendChild(listItem);
  
//       }
//     }
    
//   }
//   emotionChart.update();
// }


/**************************************************** */

// Function to show a toast message
function showToast(message) {
  // Create toast element
  var toast = document.createElement("div");
  toast.classList.add("toast");
  toast.textContent = message;

  // Add toast to the container
  var toastContainer = document.getElementById("toastContainer");
  toastContainer.appendChild(toast);

  // Show toast
  setTimeout(() => {
    toast.style.display = "block";
    toast.style.opacity = 1;
    // toast.style.bottom = "20px"; // Raise the toast
  }, 100); // Small timeout to allow for CSS transition

  // Hide and remove toast after 3 seconds
  setTimeout(() => {
    toast.style.opacity = 0;
    toast.style.bottom = "0px"; // Lower the toast

    // Remove element after fade out
    setTimeout(() => {
      toastContainer.removeChild(toast);
    }, 500); // Matches the CSS transition
  }, 3000);
}



function generateSessionId() {
  const timestamp = Date.now().toString(36); // Convert timestamp to base-36 for compactness
  const randomString = Math.random().toString(36).substring(2, 15); // Generate a random string
  return timestamp + '_' + randomString;
}


// speech.js

// -------------------- Speech Recognition Functionality --------------------
let recordingOngoing = false;
const recordButton = document.getElementById('recordButton');
const chatInput = document.getElementById('chatInput');

if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const recognition = new SpeechRecognition();

  recognition.lang = 'en-US';
  recognition.continuous = false;
  recognition.interimResults = false;

  recognition.onstart = () => {
    console.log('Speech recognition started');
    recordingOngoing = true;
    recordButton.innerHTML = '<i class="fas fa-spinner fa-spin has-text-link"></i>';
    chatInput.value = '';
    chatInput.placeholder = 'Listening...';
  };

  recognition.onspeechend = () => {
    console.log('Speech recognition ended');
    recognition.stop();
    recordingOngoing = false;
    recordButton.innerHTML = '<i class="fas fa-microphone has-text-link"></i>';
    chatInput.placeholder = 'Type message or press CTRL+R to record...';
    chatInput.focus();
  };

  recognition.onerror = (event) => {
    console.error('Speech recognition error:', event.error);
    recordingOngoing = false;
    recordButton.innerHTML = '<i class="fas fa-microphone has-text-link"></i>';
    chatInput.placeholder = 'Type message or press CTRL+R to record...';
  };

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    console.log('Transcript:', transcript);
    chatInput.value = transcript;
    recordButton.innerHTML = '<i class="fas fa-microphone has-text-link"></i>';
    chatInput.placeholder = 'Type message or press CTRL+R to record...';
    recordingOngoing = false;
  };

  // Event listener for the record button
  recordButton.addEventListener('click', () => {
    if (!recordingOngoing) {
      console.log('Starting speech recognition');
      recognition.start();
    } else {
      console.log('Stopping speech recognition');
      recognition.stop();
    }
  });

  // Event listenner for key "CTRL+R" to start/stop recording
  document.addEventListener('keydown', (event) => {
    if ((event.ctrlKey && event.key === 'r') || (event.ctrlKey && event.key === 'R')) {
      if (!recordingOngoing) {
        event.preventDefault();
        console.log('Starting speech recognition');
        recognition.start();
      } else {
        event.preventDefault();
        console.log('Stopping speech recognition');
        recognition.stop();
      }
    }
  });
} else {
  console.error("Speech recognition is not supported in this browser.");
  document.getElementById('result').textContent = "Speech recognition is not supported in this browser.";
}
let currentTamilVoiceText = "";

// Fixed Image Upload Logic
async function uploadImage(input) {
    const file = input.files[0];
    if (!file) return;

    const scanBtn = document.querySelector('.btn-gold');
    const originalBtnText = scanBtn.innerHTML;
    scanBtn.innerHTML = "⚡ Analyzing...";

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict', { method: 'POST', body: formData });
        const data = await response.json();
        const p = data.protocol;

        // Populate English Lists
        const fill = (id, items) => {
            document.getElementById(id).innerHTML = items.map(i => `<li>${i}</li>`).join("");
        };
        fill('p-about', p.about);
        fill('p-cause', p.cause);
        fill('p-prevention', p.prevention);
        fill('p-treatment', p.treatment);

        // Store unified Tamil script for Voice button
        currentTamilVoiceText = p.tamil_voice;

        // Update UI Visuals
        document.getElementById('resCond').innerText = data.condition;
        document.getElementById('resAcc').innerText = data.accuracy;
        document.getElementById('heatmapImg').src = data.heatmap + "?v=" + Date.now();
        document.getElementById('resultsArea').style.display = 'block';

    } catch (error) {
        console.error("Upload failed:", error);
        alert("Server error. Please check your backend.");
    } finally {
        scanBtn.innerHTML = originalBtnText;
    }
}

// Fixed Voice Functionality
function playVoice() {
    if (!currentTamilVoiceText) return;
    window.speechSynthesis.cancel();
    
    const utter = new SpeechSynthesisUtterance(currentTamilVoiceText);
    const voices = window.speechSynthesis.getVoices();
    
    // Specifically search for a Tamil engine
    const taVoice = voices.find(v => v.lang.includes('ta-IN') || v.lang === 'ta');
    if (taVoice) utter.voice = taVoice;
    
    utter.lang = 'ta-IN';
    utter.rate = 0.85; 
    window.speechSynthesis.speak(utter);
}

// Essential for mobile browsers to pre-load voice engines
window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
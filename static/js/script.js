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

// ============ SEASONAL ADVISORY FUNCTIONS ============

function startSeasonalAdvisory() {
    document.getElementById('landing-view').style.display = 'none';
    document.getElementById('seasonal-view').style.display = 'block';
    document.getElementById('app-view').style.display = 'none';
}

async function getSeasonalAdvice() {
    const crop = document.getElementById('cropSelect').value;
    const season = document.getElementById('seasonSelect').value;
    const region = document.getElementById('regionInput').value;

    if (!crop || !season) {
        alert('Please select crop type and season');
        return;
    }

    // Show loading state
    document.getElementById('seasonal-form-view').style.display = 'none';
    document.getElementById('seasonal-results-view').style.display = 'block';
    document.getElementById('diseasesContainer').innerHTML = '<p style="text-align: center; color: #64748b;">Loading disease predictions...</p>';

    try {
        const response = await fetch('/seasonal-advisory', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ crop, season, region })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to get predictions');
        }

        // Display disease risks
        let diseaseHTML = '';
        data.diseases.forEach(disease => {
            const riskColor = disease.risk === 'High' ? '#ef4444' : disease.risk === 'Medium' ? '#f59e0b' : '#10b981';
            const riskBg = disease.risk === 'High' ? '#fef2f2' : disease.risk === 'Medium' ? '#fffbeb' : '#f0fdf4';
            diseaseHTML += `
                <div style="background: ${riskBg}; padding: 12px; border-radius: 8px; border-left: 4px solid ${riskColor};">
                    <div style="font-weight: 600; color: ${riskColor}; display: flex; justify-content: space-between; align-items: center;">
                        <span>${disease.name}</span>
                        <span style="font-size: 12px; background: ${riskColor}; color: white; padding: 4px 8px; border-radius: 4px;">${disease.risk}</span>
                    </div>
                    <p style="color: var(--text-light); font-size: 13px; margin-top: 6px;">${disease.description}</p>
                </div>
            `;
        });
        document.getElementById('diseasesContainer').innerHTML = diseaseHTML;

        // Display prevention tips
        let preventionHTML = '<ul style="margin: 0; padding-left: 20px;">';
        data.prevention.forEach(tip => {
            preventionHTML += `<li style="margin: 8px 0; color: var(--text-dark); font-size: 14px;">${tip}</li>`;
        });
        preventionHTML += '</ul>';
        document.getElementById('preventionContainer').innerHTML = preventionHTML;

    } catch (error) {
        console.error('Error:', error);
        alert('Error: ' + error.message);
        document.getElementById('seasonal-form-view').style.display = 'block';
        document.getElementById('seasonal-results-view').style.display = 'none';
    }
}

function resetSeasonalForm() {
    document.getElementById('cropSelect').value = '';
    document.getElementById('seasonSelect').value = '';
    document.getElementById('regionInput').value = '';
    document.getElementById('seasonal-form-view').style.display = 'block';
    document.getElementById('seasonal-results-view').style.display = 'none';
}

// Navigation function (update existing one if present)
function goBack() {
    const seasonalView = document.getElementById('seasonal-view');
    const appView = document.getElementById('app-view');
    const landingView = document.getElementById('landing-view');

    if (seasonalView && seasonalView.style.display !== 'none') {
        seasonalView.style.display = 'none';
        landingView.style.display = 'block';
    } else if (appView && appView.style.display !== 'none') {
        appView.style.display = 'none';
        landingView.style.display = 'block';
    }
}

function startApp() {
    document.getElementById('landing-view').style.display = 'none';
    document.getElementById('seasonal-view').style.display = 'none';
    document.getElementById('app-view').style.display = 'block';
}
class ArabicAudioRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.stream = null;
        this.recordingStartTime = null;
        this.maxRecordingTime = 120000; // 2 minutes max
        this.recordingTimer = null;
    }

    async initialize() {
        try {
            // Request microphone access
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    sampleRate: 16000 // Optimal for speech recognition
                } 
            });
            
            // Create MediaRecorder
            this.mediaRecorder = new MediaRecorder(this.stream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            // Set up event handlers
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                this.processRecording();
            };
            
            console.log('🎤 Audio recorder initialized successfully');
            return true;
            
        } catch (error) {
            console.error('❌ Error initializing audio recorder:', error);
            this.showError('لا يمكن الوصول إلى الميكروفون. يرجى السماح بالوصول للميكروفون.');
            return false;
        }
    }

    startRecording() {
        if (!this.mediaRecorder || this.isRecording) {
            return false;
        }

        try {
            this.audioChunks = [];
            this.recordingStartTime = Date.now();
            this.mediaRecorder.start(1000); // Collect data every second
            this.isRecording = true;
            
            // Start recording timer
            this.startRecordingTimer();
            
            // Update UI
            this.updateRecordingUI(true);
            
            console.log('🎤 Recording started');
            return true;
            
        } catch (error) {
            console.error('❌ Error starting recording:', error);
            this.showError('خطأ في بدء التسجيل');
            return false;
        }
    }

    stopRecording() {
        if (!this.mediaRecorder || !this.isRecording) {
            return false;
        }

        try {
            this.mediaRecorder.stop();
            this.isRecording = false;
            
            // Stop recording timer
            this.stopRecordingTimer();
            
            // Update UI
            this.updateRecordingUI(false);
            
            console.log('🛑 Recording stopped');
            return true;
            
        } catch (error) {
            console.error('❌ Error stopping recording:', error);
            this.showError('خطأ في إيقاف التسجيل');
            return false;
        }
    }

    startRecordingTimer() {
        let seconds = 0;
        this.recordingTimer = setInterval(() => {
            seconds++;
            this.updateTimerDisplay(seconds);
            
            // Auto-stop after max time
            if (seconds * 1000 >= this.maxRecordingTime) {
                this.stopRecording();
                this.showMessage('تم إيقاف التسجيل تلقائياً بعد دقيقتين');
            }
        }, 1000);
    }

    stopRecordingTimer() {
        if (this.recordingTimer) {
            clearInterval(this.recordingTimer);
            this.recordingTimer = null;
        }
    }

    updateTimerDisplay(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        const timeString = `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
        
        const timerElement = document.getElementById('recordingTimer');
        if (timerElement) {
            timerElement.textContent = timeString;
        }
    }

    updateRecordingUI(isRecording) {
        const recordBtn = document.getElementById('recordButton');
        const stopBtn = document.getElementById('stopButton');
        const recordingIndicator = document.getElementById('recordingIndicator');
        const timerElement = document.getElementById('recordingTimer');
        
        if (recordBtn) {
            recordBtn.disabled = isRecording;
            recordBtn.textContent = isRecording ? 'جاري التسجيل...' : '🎤 ابدأ التسجيل';
            recordBtn.className = isRecording ? 'btn recording' : 'btn primary-btn';
        }
        
        if (stopBtn) {
            stopBtn.disabled = !isRecording;
            stopBtn.style.display = isRecording ? 'inline-block' : 'none';
        }
        
        if (recordingIndicator) {
            recordingIndicator.style.display = isRecording ? 'block' : 'none';
        }
        
        if (timerElement) {
            timerElement.style.display = isRecording ? 'block' : 'none';
            if (!isRecording) {
                timerElement.textContent = '0:00';
            }
        }
    }

    async processRecording() {
        if (this.audioChunks.length === 0) {
            this.showError('لم يتم تسجيل أي صوت');
            return;
        }

        try {
            // Create audio blob
            const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
            const recordingDuration = (Date.now() - this.recordingStartTime) / 1000;
            
            console.log(`🎵 Audio recorded: ${recordingDuration}s, ${audioBlob.size} bytes`);
            
            // Show processing message
            this.showMessage('جاري معالجة التسجيل الصوتي...');
            
            // Send to server for processing
            await this.sendAudioForAssessment(audioBlob, recordingDuration);
            
        } catch (error) {
            console.error('❌ Error processing recording:', error);
            this.showError('خطأ في معالجة التسجيل الصوتي');
        }
    }

    async sendAudioForAssessment(audioBlob, duration) {
        try {
            // Get current student and text IDs
            const studentId = document.getElementById('currentStudentId')?.value;
            const textId = document.getElementById('currentTextId')?.value;
            
            if (!studentId || !textId) {
                this.showError('يرجى اختيار الطالب والنص أولاً');
                return;
            }
            
            // Create form data
            const formData = new FormData();
            formData.append('student_id', studentId);
            formData.append('text_id', textId);
            formData.append('audio', audioBlob, 'recording.webm');
            formData.append('duration', duration.toString());
            
            // Send to assessment API
            const response = await fetch('/api/assessments', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showSuccess('تم تقييم القراءة بنجاح! 🎉');
                this.displayAssessmentResults(result);
            } else {
                this.showError(`خطأ في التقييم: ${result.message}`);
            }
            
        } catch (error) {
            console.error('❌ Error sending audio for assessment:', error);
            this.showError('خطأ في إرسال التسجيل للتقييم');
        }
    }

    displayAssessmentResults(result) {
        if (result.result && result.result.scores) {
            const scores = result.result.scores;
            
            // Update score displays
            document.getElementById('overallScore').textContent = Math.round(scores.overall || 0);
            document.getElementById('pronunciationScore').textContent = Math.round(scores.pronunciation || 0);
            document.getElementById('fluencyScore').textContent = Math.round(scores.fluency || 0);
            document.getElementById('accuracyScore').textContent = Math.round(scores.accuracy || 0);
            document.getElementById('comprehensionScore').textContent = Math.round(scores.comprehension || 0);
            
            // Show results section
            document.getElementById('scoresDisplay').style.display = 'block';
            document.getElementById('resultDisplay').style.display = 'block';
            
            // Display detailed feedback
            if (result.result.feedback) {
                document.getElementById('feedbackText').textContent = result.result.feedback;
            }
            
            // Scroll to results
            document.getElementById('scoresDisplay').scrollIntoView({ behavior: 'smooth' });
        }
    }

    showMessage(message) {
        this.showNotification(message, 'info');
    }

    showSuccess(message) {
        this.showNotification(message, 'success');
    }

    showError(message) {
        this.showNotification(message, 'error');
    }

    showNotification(message, type) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 5px;
            color: white;
            font-weight: bold;
            z-index: 1000;
            max-width: 300px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        `;
        
        // Set background color based on type
        switch (type) {
            case 'success':
                notification.style.backgroundColor = '#28a745';
                break;
            case 'error':
                notification.style.backgroundColor = '#dc3545';
                break;
            default:
                notification.style.backgroundColor = '#007bff';
        }
        
        // Add to page
        document.body.appendChild(notification);
        
        // Remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
    }

    cleanup() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }
        if (this.recordingTimer) {
            clearInterval(this.recordingTimer);
        }
    }
}

// Global recorder instance
let audioRecorder = null;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', async () => {
    audioRecorder = new ArabicAudioRecorder();
    
    // Try to initialize (will ask for microphone permission)
    const initialized = await audioRecorder.initialize();
    
    if (initialized) {
        console.log('✅ Audio recorder ready');
    } else {
        console.log('❌ Audio recorder initialization failed');
    }
});

// Cleanup when page unloads
window.addEventListener('beforeunload', () => {
    if (audioRecorder) {
        audioRecorder.cleanup();
    }
});

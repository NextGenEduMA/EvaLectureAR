// Real-time Arabic Audio Recorder with WebSocket
class RealTimeRecorder {
    constructor() {
        this.socket = null;
        this.mediaRecorder = null;
        this.audioStream = null;
        this.isRecording = false;
        this.recordingStartTime = null;
        this.recordingTimer = null;
        
        this.initializeSocket();
    }

    initializeSocket() {
        // Initialize Socket.IO connection
        this.socket = io();
        
        // Socket event handlers
        this.socket.on('connected', (data) => {
            console.log('✅ Connected to server:', data.message);
            this.showNotification(data.message, 'success');
        });
        
        this.socket.on('recording_started', (data) => {
            console.log('🎤 Recording started:', data);
            this.onRecordingStarted(data);
        });
        
        this.socket.on('chunk_processed', (data) => {
            // Update UI with chunk count (optional)
            this.updateChunkCounter(data.chunk_count);
        });
        
        this.socket.on('realtime_feedback', (data) => {
            this.showRealtimeFeedback(data);
        });
        
        this.socket.on('processing_started', (data) => {
            this.showNotification(data.message, 'info');
            this.showProcessingUI();
        });
        
        this.socket.on('assessment_complete', (data) => {
            console.log('✅ Assessment complete:', data);
            this.onAssessmentComplete(data);
        });
        
        this.socket.on('recording_error', (data) => {
            this.showNotification(data.message, 'error');
            this.resetRecordingUI();
        });
        
        this.socket.on('assessment_error', (data) => {
            this.showNotification(data.message, 'error');
            this.resetRecordingUI();
        });
    }

    async startRecording() {
        try {
            // Get student and text IDs
            const studentId = document.getElementById('currentStudentId')?.value;
            const textId = document.getElementById('currentTextId')?.value;
            
            if (!studentId || !textId) {
                this.showNotification('يرجى اختيار الطالب والنص أولاً', 'error');
                return;
            }
            
            // Request microphone access
            this.audioStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    sampleRate: 16000
                }
            });
            
            // Create MediaRecorder
            this.mediaRecorder = new MediaRecorder(this.audioStream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            // Handle audio data
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0 && this.isRecording) {
                    this.sendAudioChunk(event.data);
                }
            };
            
            // Start recording
            this.mediaRecorder.start(500); // Send chunks every 500ms
            this.isRecording = true;
            this.recordingStartTime = Date.now();
            
            // Start recording session on server
            this.socket.emit('start_recording', {
                student_id: parseInt(studentId),
                text_id: parseInt(textId)
            });
            
            // Start UI timer
            this.startRecordingTimer();
            
        } catch (error) {
            console.error('❌ Error starting recording:', error);
            this.showNotification('خطأ في الوصول للميكروفون', 'error');
        }
    }

    stopRecording() {
        if (!this.isRecording) return;
        
        try {
            // Stop MediaRecorder
            this.mediaRecorder.stop();
            this.isRecording = false;
            
            // Stop audio stream
            if (this.audioStream) {
                this.audioStream.getTracks().forEach(track => track.stop());
            }
            
            // Stop timer
            this.stopRecordingTimer();
            
            // Notify server
            this.socket.emit('stop_recording');
            
            // Update UI
            this.updateRecordingUI(false);
            
        } catch (error) {
            console.error('❌ Error stopping recording:', error);
            this.showNotification('خطأ في إيقاف التسجيل', 'error');
        }
    }

    async sendAudioChunk(audioBlob) {
        try {
            // Convert blob to base64
            const arrayBuffer = await audioBlob.arrayBuffer();
            const base64Audio = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
            
            // Send to server
            this.socket.emit('audio_chunk', {
                audio: base64Audio
            });
            
        } catch (error) {
            console.error('❌ Error sending audio chunk:', error);
        }
    }

    onRecordingStarted(data) {
        this.updateRecordingUI(true);
        this.showSessionInfo(data.session_info);
        this.showNotification(data.message, 'success');
    }

    onAssessmentComplete(data) {
        this.resetRecordingUI();
        this.showNotification(data.message, 'success');
        
        // Call the window function if it exists (for custom handling)
        if (typeof window.onAssessmentComplete === 'function') {
            window.onAssessmentComplete(data.result);
        } else if (data.result && data.result.scores) {
            this.displayAssessmentResults(data.result);
        }
    }

    showSessionInfo(sessionInfo) {
        const infoElement = document.getElementById('sessionInfo');
        if (infoElement) {
            infoElement.innerHTML = `
                <div class="session-info">
                    <h4>معلومات الجلسة</h4>
                    <p><strong>الطالب:</strong> ${sessionInfo.student_name}</p>
                    <p><strong>النص:</strong> ${sessionInfo.text_title}</p>
                </div>
            `;
            infoElement.style.display = 'block';
        }
    }

    showRealtimeFeedback(feedback) {
        const feedbackElement = document.getElementById('realtimeFeedback');
        if (feedbackElement) {
            feedbackElement.textContent = feedback.message;
            feedbackElement.className = 'realtime-feedback';
            
            if (feedback.volume_warning) {
                feedbackElement.className += ' warning';
            } else if (feedback.volume_ok) {
                feedbackElement.className += ' success';
            }
            
            feedbackElement.style.display = 'block';
            
            // Hide after 3 seconds
            setTimeout(() => {
                feedbackElement.style.display = 'none';
            }, 3000);
        }
    }

    updateChunkCounter(count) {
        const counterElement = document.getElementById('chunkCounter');
        if (counterElement) {
            counterElement.textContent = `معالج: ${count} قطعة صوتية`;
        }
    }

    startRecordingTimer() {
        let seconds = 0;
        this.recordingTimer = setInterval(() => {
            seconds++;
            this.updateTimerDisplay(seconds);
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
            recordBtn.textContent = isRecording ? 'جاري التسجيل...' : '🎤 ابدأ التسجيل المباشر';
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

    showProcessingUI() {
        const processingElement = document.getElementById('processingIndicator');
        if (processingElement) {
            processingElement.style.display = 'block';
            processingElement.innerHTML = `
                <div class="processing-animation">
                    <div class="spinner"></div>
                    <p>جاري تحليل القراءة وإنشاء التقييم...</p>
                </div>
            `;
        }
    }

    resetRecordingUI() {
        this.updateRecordingUI(false);
        
        const processingElement = document.getElementById('processingIndicator');
        if (processingElement) {
            processingElement.style.display = 'none';
        }
        
        const sessionInfoElement = document.getElementById('sessionInfo');
        if (sessionInfoElement) {
            sessionInfoElement.style.display = 'none';
        }
    }

    displayAssessmentResults(result) {
        if (result.scores) {
            const scores = result.scores;
            
            // Update score displays
            document.getElementById('overallScore').textContent = Math.round(scores.overall || 0);
            document.getElementById('pronunciationScore').textContent = Math.round(scores.pronunciation || 0);
            document.getElementById('fluencyScore').textContent = Math.round(scores.fluency || 0);
            document.getElementById('accuracyScore').textContent = Math.round(scores.accuracy || 0);
            document.getElementById('comprehensionScore').textContent = Math.round(scores.comprehension || 0);
            
            // Show results section
            document.getElementById('scoresDisplay').style.display = 'block';
            
            // Display detailed feedback
            if (result.feedback) {
                const feedbackElement = document.getElementById('feedbackText');
                if (feedbackElement) {
                    feedbackElement.textContent = result.feedback;
                }
            }
            
            // Scroll to results
            document.getElementById('scoresDisplay').scrollIntoView({ behavior: 'smooth' });
        }
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
        if (this.audioStream) {
            this.audioStream.getTracks().forEach(track => track.stop());
        }
        if (this.recordingTimer) {
            clearInterval(this.recordingTimer);
        }
        if (this.socket) {
            this.socket.disconnect();
        }
    }
}

// Global recorder instance
let realtimeRecorder = null;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    realtimeRecorder = new RealTimeRecorder();
    console.log('✅ Real-time recorder initialized');
});

// Cleanup when page unloads
window.addEventListener('beforeunload', () => {
    if (realtimeRecorder) {
        realtimeRecorder.cleanup();
    }
});

// Global functions for buttons
function startRealTimeRecording() {
    if (realtimeRecorder) {
        realtimeRecorder.startRecording();
    }
}

function stopRealTimeRecording() {
    if (realtimeRecorder) {
        realtimeRecorder.stopRecording();
    }
}

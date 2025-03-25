<template>
  <div class="container">
    <h1 class="title">EMOTION INTERPRETATION FROM SERMON</h1>

    <!-- File Upload Section -->
    <div class="upload-section">
      <h3>Upload a Video</h3>
      <FileUpload @file-uploaded="handleFileUpload" />
      <p v-if="message" :class="{'error-message': isError}">{{ message }}</p>
    </div>

    <!-- Uploaded Video Section -->
    <div v-if="uploadedVideo" class="video-section">
      <h3>Uploaded Video</h3>
      <video :src="uploadedVideo" controls class="small-video" @timeupdate="checkForFeedback" ref="videoPlayer"></video>
    </div>

    <!-- Processing Progress Bar -->
    <div v-if="processing" class="progress-container">
      <p>{{ progressStatus }}</p>
      <div class="progress-bar" :style="{ width: processingProgress + '%' }"></div>
      <p>{{ processingProgress }}% completed | Please wait...</p>
    </div>
    
    <div class="progress-bar" 
     :class="{ green: processingProgress === 100 }" 
     :style="{ width: processingProgress + '%' }">
    </div>

    <!-- Summary Information Row -->
    <div v-if="dressCode || eyesInCameraPercent !== null" class="summary-row">
      <!-- Dress Code Display -->
      <div v-if="dressCode" class="dress-code">
        <span class="label">Dress :  </span> 
        <span :class="dressCode.toLowerCase()">{{ dressCode }}</span>
      </div>

      <!-- Eye Contact Percentage Button -->
      <button v-if="eyesInCameraPercent !== null" @click="toggleEyeDetails" class="btn eye-percentage-btn">
        Eye Contact: {{ eyesInCameraPercent.toFixed(2) }}%
      </button>

      <!-- Feedback Button -->
      <button v-if="chunkFeedback.length > 0" @click="showFirstFeedback" class="btn feedback-button">
        Show Feedback
      </button>

      <!-- Show Details Button -->
      <button v-if="showDetailsButton" @click="toggleDetails" class="btn details-button">
        {{ showDetails ? 'Hide Details' : 'Show Details' }}
      </button>
    </div>

    <!-- 20-Second Chunk Feedback Section -->
    <div v-if="showFeedback" class="feedback-section">
      <div class="feedback-header">
        <h3>Feedback for {{ currentFeedback.start_time }}s - {{ currentFeedback.end_time }}s</h3>
        <button class="close-btn" @click="closeFeedback">&times;</button>
      </div>
      
      <div class="feedback-grid">
        <div class="feedback-item">
          <h4>Live Transcription</h4>
          <p>{{ currentFeedback.transcript || "No speech detected" }}</p>
        </div>
        
        <div class="feedback-item">
          <h4>Eye Contact</h4>
          <div class="eye-contact-meter">
            <div class="meter-bar" :style="{ width: currentFeedback.eye_contact_percent + '%' }"></div>
            <span>{{ currentFeedback.eye_contact_percent }}%</span>
          </div>
        </div>
        
        <div class="feedback-item">
          <h4>Dominant Emotions</h4>
          <div class="emotion-display">
            <div class="emotion-pair">
              <span class="emotion-label">Video:</span>
              <span class="emotion-value" :class="currentFeedback.dominant_video_emotion.toLowerCase()">
                {{ currentFeedback.dominant_video_emotion }}
              </span>
            </div>
            <div class="emotion-pair">
              <span class="emotion-label">Audio:</span>
              <span class="emotion-value" :class="currentFeedback.dominant_audio_emotion.toLowerCase()">
                {{ currentFeedback.dominant_audio_emotion }}
              </span>
            </div>
            <div class="emotion-pair">
              <span class="emotion-label">Text:</span>
              <span class="emotion-value" :class="currentFeedback.dominant_text_emotion.toLowerCase()">
                {{ currentFeedback.dominant_text_emotion }}
              </span>
            </div>
          </div>
        </div>
        
        <div class="feedback-item">
          <h4>Emotion Match</h4>
          <div class="match-indicator" :class="{ 'match': currentFeedback.emotion_match, 'no-match': !currentFeedback.emotion_match }">
            {{ currentFeedback.emotion_match ? 'POSITIVE' : 'NEEDS IMPROVEMENT' }}
          </div>
          <p class="match-explanation" v-if="!currentFeedback.emotion_match">
            Emotions don't match across modalities
          </p>
        </div>
      </div>
      
      <button class="btn next-feedback-btn" @click="showNextFeedback" v-if="hasMoreFeedback">
        Next Segment ({{ nextFeedbackStartTime }}s)
      </button>
    </div>

    <!-- Eye Contact Details Section -->
    <div v-if="showEyeDetails && graphs" class="eye-details-section">
      <div class="eye-percentages">
        <div class="eyes-percentage">
          <span class="label">Eyes in Camera : </span> 
          <span class="percentage">{{ eyesInCameraPercent.toFixed(2) }}%</span>
        </div>
        <div class="eyes-percentage">
          <span class="label">Eyes Not in Camera : </span> 
          <span class="percentage">{{ eyesNotInCameraPercent.toFixed(2) }}%</span>
        </div>
      </div>
      
      <!-- Eye Contact Graph -->
      <div v-if="graphs.eye_graph" class="graph-container">
        <h4>EYE CONTACT WITH CAMERA OVER TIME</h4>
        <img :src="`http://127.0.0.1:5000/graphs/${graphs.eye_graph}`" alt="Eye Contact Graph" class="emotion-graph" />
      </div>
    </div>

    <!-- Results Sections (Conditionally Shown) -->
    <div v-if="showDetails">
      <!-- Graphs Section -->
      <div v-if="graphs" class="graph-section">
        <!-- eslint-disable-next-line vue/no-use-v-if-with-v-for -->
        <template v-for="(graph, key) in graphs">
          <div v-if="key !== 'eye_graph'" :key="key" class="graph-container">
            <h4>{{ key.replace('_', ' ').toUpperCase() }} GRAPH</h4>
            <img :src="`http://127.0.0.1:5000/graphs/${graph}`" alt="Graph" class="emotion-graph" />
          </div>
        </template>
      </div>    
    
      <div class="download-buttons" v-if="graphs">
        <a :href="`http://127.0.0.1:5000/download_csv/${csv}`" class="btn" download>Download CSV</a>
      </div>
      
      <!-- Video Transcript -->
      <div v-if="transcriptText && transcriptText.length > 0" class="transcript-container">
        <h3>Video Transcript</h3>
        <textarea v-model="transcriptText" rows="6" class="transcript-editor"></textarea>
        <button @click="submitTranscript" class="btn">Update Transcript</button>
      </div>

      <!-- Frame-wise Emotions Table -->
      <div v-if="videoEmotions.length > 0" class="frames-container">
        <h4>Frame-wise Video, Audio, and Text Emotions</h4>
        <table class="emotion-table">
          <thead>
            <tr>
              <th>Time (s)</th>
              <th>Video Emotion</th>
              <th>Audio Emotion</th>
              <th>Text Emotion</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(item, index) in videoEmotions" :key="index">
              <td>{{ item.Time }}</td>
              <td>{{ item['Video Emotion'] || 'Neutral' }}</td>
              <td>{{ audioEmotions[index]?.['Audio Emotion'] || 'Neutral' }}</td>
              <td>{{ textEmotions[index]?.['Text Emotion'] || 'Neutral' }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<script>
import FileUpload from './components/FileUpload.vue';

export default {
  components: {
    FileUpload,
  },
  data() {
    return {
      message: "",
      isError: false,
      uploadedVideo: null,
      frames: [],
      videoEmotions: [],
      audioEmotions: [],
      textEmotions: [],
      transcript: [],
      transcriptText: "",
      dressCode: null,
      graphs: null,
      csv: "",
      loading: false,
      matchingEmotions: [],
      processing: false,
      processingProgress: 0, 
      selectedFilter: "all", 
      filteredGraph: "",
      progressStatus: "idle",
      progressStep: "",
      progress: 0,
      allGraph: "",
      highlightGraph: "",
      fadeGraph: "",
      suddenChangesGraph: "",
      eyesInCameraPercent: null,
      eyesNotInCameraPercent: null, 
      showDetails: false,
      showDetailsButton: false,
      showEyeDetails: false,
      
      // Feedback system variables
      showFeedback: false,
      chunkFeedback: [],
      currentFeedbackIndex: 0,
      currentFeedback: {},
      videoDuration: 0,
      feedbackShownForChunks: []
    };
  },
  computed: {
    hasMoreFeedback() {
      return this.currentFeedbackIndex < this.chunkFeedback.length - 1;
    },
    nextFeedbackStartTime() {
      if (this.hasMoreFeedback) {
        return this.chunkFeedback[this.currentFeedbackIndex + 1].start_time;
      }
      return 0;
    }
  },
  methods: {
    uploadVideo(event) {
      this.file = event.target.files[0];
    },
    handleFileUpload(file) {
      this.uploadedVideo = URL.createObjectURL(file);
      this.uploadFile(file);
    },
    async updateGraph() {
      try {
        const response = await fetch('http://127.0.0.1:5000/generate_graph?filter=${this.selectedFilter}');
        const data = await response.json();
        this.graph = data.graph;
      } catch (error) {
        console.error("Error fetching filtered graph:", error);
      }
    },

    async fetchGraph() {
      try {
        const response = await fetch('http://127.0.0.1:5000/generate_graph?filter=${this.selectedFilter}');
        if (!response.ok) {
          throw new Error("Failed to fetch the graph.");
        }
        const data = await response.json();
        this.graph = data.graph;
      } catch (error) {
        console.error(error);
        this.message = "Error fetching the graph.";
        this.isError = true;
      }
    },

    async uploadFile(file) {
      this.uploading = true;
      this.isError = false;
      this.processing = true;
      this.progressStatus = "Uploading...";
      this.processingProgress = 0;

      const formData = new FormData();
      formData.append("file", file);

      try {
        const pollProgress = setInterval(async () => {
          const response = await fetch("http://127.0.0.1:5000/progress");
          if (response.ok) {
            const progressData = await response.json();
            this.progressStatus = progressData.step;
            this.processingProgress = progressData.progress;

            if (progressData.progress >= 100) {
              clearInterval(pollProgress);
            }
          }
        }, 500);

        const response = await fetch("http://127.0.0.1:5000/upload", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error("Failed to process the video. Please try again.");
        }

        const data = await response.json();
        this.processing = false;

        console.log("🔹 Response from Backend:", data);

        if (data.dress_code) {
          this.dressCode = data.dress_code;
          console.log("Detected Dress Code: ", this.dressCode);
        } else {
          this.dressCode = "Unknown";
        }

        this.eyesInCameraPercent = data.eyes_in_camera_percent;
        this.eyesNotInCameraPercent = data.eyes_not_in_camera_percent;

        this.graphs = data.graphs;
        this.transcriptText = data.transcript || "No transcript available.";
        this.videoEmotions = data.video_emotions || [];
        this.audioEmotions = this.videoEmotions.map(item => ({
          "Audio Emotion": item["Audio Emotion"] || "Neutral"
        }));
        this.textEmotions = this.videoEmotions.map(item => ({
          "Text Emotion": item["Text Emotion"] || "Neutral"
        }));

        this.graph = data.graph;
        this.csv = data.csv;
        this.message = "Video processing completed!";
        if (this.graphs) {
          this.message = "Graphs plotted. You can now download the CSV.";
          this.showDetailsButton = true;
        } else {
          this.message = "Processing completed without graphs.";
        }

        // Store chunk feedback data
        if (data.chunk_feedback && data.chunk_feedback.length > 0) {
          this.chunkFeedback = data.chunk_feedback;
          this.videoDuration = data.video_duration;
        }
      } catch (error) {
        console.error(error);
        this.message = error.message || "Error uploading file. Please try again.";
        this.isError = true;
      } finally {
        this.processing = false;
        this.uploading = false;
      }
    },

    // Feedback system methods
    showFirstFeedback() {
      if (this.chunkFeedback.length > 0) {
        this.currentFeedbackIndex = 0;
        this.currentFeedback = this.chunkFeedback[0];
        this.showFeedback = true;
        this.feedbackShownForChunks = [0];
      }
    },

    showNextFeedback() {
      if (this.hasMoreFeedback) {
        this.currentFeedbackIndex++;
        this.currentFeedback = this.chunkFeedback[this.currentFeedbackIndex];
        this.feedbackShownForChunks.push(this.currentFeedbackIndex);
      } else {
        this.closeFeedback();
      }
    },

    closeFeedback() {
      this.showFeedback = false;
    },

    checkForFeedback() {
      if (!this.chunkFeedback.length || !this.$refs.videoPlayer) return;
      
      const currentTime = this.$refs.videoPlayer.currentTime;
      const currentChunk = this.chunkFeedback.findIndex(
        chunk => currentTime >= chunk.start_time && currentTime < chunk.end_time
      );
      
      if (currentChunk !== -1) {
        // Only update if we're moving to a new chunk
        if (this.currentFeedbackIndex !== currentChunk) {
          this.currentFeedbackIndex = currentChunk;
          this.currentFeedback = this.chunkFeedback[currentChunk];
          this.showFeedbackModal = true;
        }
      }
    },

    toggleDetails() {
      this.showDetails = !this.showDetails;
    },

    toggleEyeDetails() {
      this.showEyeDetails = !this.showEyeDetails;
    },

    async submitTranscript() {
      const response = await fetch("http://127.0.0.1:5000/submit_script", {
        method: "POST",
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ script: this.transcriptText })
      });

      if (response.ok) {
        const data = await response.json();
        this.textEmotions = data.emotions || [];
      } else {
        console.error('Failed to analyze transcript emotions.');
      }
    },
    async fetchMatchingEmotions() {
      try {
        console.log("Fetching matching emotions...");
        const response = await fetch("http://127.0.0.1:5000/matching_emotions");
        if (!response.ok) {
          throw new Error("Failed to fetch matching emotions.");
        }
        this.matchingEmotions = await response.json();
        console.log("Matching Emotions:", this.matchingEmotions);
      } catch (error) {
        console.error(error);
        this.message = "BACKEND IS NOT RUNNING";
        this.isError = true;
      }
    },

    async fetchProgress() {
    const interval = setInterval(async () => {
      const response = await fetch("http://127.0.0.1:5000/progress");
      const data = await response.json();
      this.progressStatus = data.status;
      this.progressStep = data.step;
      this.progress = data.progress;
      if (data.status === "completed") clearInterval(interval);
    }, 1000);
    },
    async fetchAllGraphs() {
      try {
          const response = await fetch(`http://127.0.0.1:5000/generate_graphs`);
          const data = await response.json();
          this.allGraph = data.all;
          this.highlightGraph = data.highlight;
          this.fadeGraph = data.fade;
          this.suddenChangesGraph = data.sudden;
      } catch (error) {
          console.error("Error fetching graphs:", error);
      }
    },
    updateFilteredGraph() {
      this.fetchAllGraphs();
    },
    async processVideo() {
      if (!this.file) {
        alert("Please upload a video.");
        return;
      }

      let formData = new FormData();
      formData.append("file", this.file);

      try {
        let response = await fetch("http://127.0.0.1:5000/upload", {
          method: "POST",
          body: formData
        });
        let result = await response.json();

        if (result.dress_code) {
          this.dressCode = result.dress_code;
        } else {
          this.dressCode = "Unknown";
        }

      } catch (error) {
        console.error("Error processing video:", error);
      }
    }
  },
  mounted() {
    this.fetchMatchingEmotions();
    this.fetchAllGraphs();
  },
};
</script>

<style scoped>
.container {
  text-align: center;
  background-color: black; 
  padding: 40px;
  margin: 0; 
  width: 100vw; 
  min-height: 100vh; 
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  color: white; 
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
}

h1, h3, h4 {
  color: #b33939;
  font-weight: 600;
}

.upload-section, .video-section, .frames-container, .graph-container, .transcript-container {
  margin-bottom: 30px;
}

.error-message {
  color: red;
  font-weight: bold;
}

input[type="file"], textarea, button {
  background-color: #b33939;
  color: white;
  padding: 10px;
  border: none;
  border-radius: 5px;
  font-size: 16px;
  transition: background-color 0.3s ease;
}

input[type="file"]:hover, textarea:hover, button:hover {
  background-color: #ff5e57;
}

textarea {
  width: 80%;
  margin: 10px 0;
  padding: 10px;
  border-radius: 5px;
  border: 1px solid #b33939;
  background-color: rgba(255, 255, 255, 0.1);
  color: white;
}

.btn {
  display: inline-block;
  background-color: #b33939;
  color: white;
  padding: 10px 20px;
  border-radius: 5px;
  cursor: pointer;
  text-decoration: none;
  margin-top: 10px;
  transition: background-color 0.3s ease;
}

.btn:hover {
  background-color: #ff5e57;
}

.details-button {
  background-color: #4caf50;
  padding: 10px 20px;
  margin: 0;
}

.details-button:hover {
  background-color: #45a049;
}

.eye-percentage-btn {
  background-color: #2196F3;
  margin: 0 10px;
}

.eye-percentage-btn:hover {
  background-color: #0b7dda;
}

.feedback-button {
  background-color: #9C27B0;
  margin: 0 10px;
}

.feedback-button:hover {
  background-color: #7B1FA2;
}

.small-video, .emotion-graph {
  width: 80%;
  max-width: 800px;
  height: auto;
  margin-bottom: 20px;
  border-radius: 10px;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
}

.emotion-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 20px;
  background-color: rgba(255, 255, 255, 0.1);
  border-radius: 10px;
  overflow: hidden;
}

.emotion-table th, .emotion-table td {
  border: 1px solid #b33939;
  padding: 12px;
  text-align: center;
}

.download-buttons {
  display: flex;
  justify-content: center;
  gap: 20px;
  margin-top: 15px;
}

.download-buttons[visible] {
  visibility: visible;
}

.loading-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.loading-popup {
  background-color: white;
  padding: 20px;
  border-radius: 10px;
  text-align: center;
}

.loading-popup img {
  width: 50px;
  height: 50px;
}

.processing-container {
  width: 100%;
  background-color: #ddd;
  height: 20px;
  border-radius: 5px;
  margin: 10px 0;
  position: relative;
}

.processing-bar {
  height: 100%;
  background-color: #ff9800;
  width: 0%;
  border-radius: 5px;
  transition: width 0.2s;
}

.progress-container {
  width: 100%;
  background-color: black;
  border-radius: 5px;
  margin: 10px 0;
  position: relative;
  height: 20px;
}

.progress-bar {
  height: 100%;
  background-color: #ff9800;
  width: 0%;
  border-radius: 5px;
  transition: width 0.2s;
}

.progress-bar.green {
  background-color: #4caf50;
}

.title {
  font-size: 2.5em;
  margin-bottom: 20px;
  text-transform: uppercase;
  letter-spacing: 2px;
}

.summary-row {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 20px;
  margin: 20px 0;
  flex-wrap: wrap;
}

.dress-code {
  font-size: 1.2em;
  padding: 10px 15px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 10px;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
}

.label {
  font-weight: bold;
}

.formal {
  color: cyan;
}

.informal {
  color: orange;
}

.eyes-percentage {
  font-size: 1.2em;
  padding: 10px 15px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 10px;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
}

.percentage {
  color: cyan;
}

.eye-details-section {
  margin: 20px 0;
  padding: 20px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 10px;
  border: 1px solid #b33939;
}

.eye-percentages {
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 20px;
}

.graph-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 30px;
}

.graph-container {
  width: 90%;
  max-width: 1000px;
  padding: 20px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 10px;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
}

/* Feedback section styles */
.feedback-section {
  background: rgba(255, 255, 255, 0.05);
  border-radius: 10px;
  padding: 20px;
  margin: 20px auto;
  max-width: 900px;
  border: 1px solid #b33939;
  position: relative;
}

.feedback-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.feedback-header h3 {
  margin: 0;
  color: #b33939;
}

.close-btn {
  background: none;
  border: none;
  color: white;
  font-size: 24px;
  cursor: pointer;
  padding: 5px;
}

.feedback-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 20px;
}

.feedback-item {
  background: rgba(255, 255, 255, 0.1);
  padding: 15px;
  border-radius: 8px;
  border: 1px solid #444;
}

.feedback-item h4 {
  color: #b33939;
  margin-top: 0;
  margin-bottom: 10px;
}

.eye-contact-meter {
  background: #333;
  height: 20px;
  border-radius: 10px;
  margin: 10px 0;
  position: relative;
}

.meter-bar {
  background: linear-gradient(to right, #ff5e57, #b33939);
  height: 100%;
  border-radius: 10px;
  transition: width 0.5s ease;
}

.eye-contact-meter span {
  position: absolute;
  right: 10px;
  top: 50%;
  transform: translateY(-50%);
  color: white;
  font-weight: bold;
}

.emotion-display {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.emotion-pair {
  display: flex;
  justify-content: space-between;
}

.emotion-label {
  font-weight: bold;
}

.emotion-value {
  padding: 3px 8px;
  border-radius: 4px;
  font-weight: bold;
}

.emotion-value.neutral {
  background-color: #666;
  color: white;
}

.emotion-value.angry {
  background-color: #ff4444;
  color: white;
}

.emotion-value.happy {
  background-color: #4CAF50;
  color: white;
}

.emotion-value.sad {
  background-color: #2196F3;
  color: white;
}

.emotion-value.surprise {
  background-color: #FF9800;
  color: white;
}

.emotion-value.fear {
  background-color: #9C27B0;
  color: white;
}

.match-indicator {
  padding: 10px;
  border-radius: 5px;
  font-weight: bold;
  text-align: center;
  margin: 10px 0;
}

.match-indicator.match {
  background-color: #4CAF50;
  color: white;
}

.match-indicator.no-match {
  background-color: #f44336;
  color: white;
}

.match-explanation {
  font-size: 0.9em;
  color: #ccc;
  margin: 5px 0 0;
}

.next-feedback-btn {
  margin-top: 20px;
  background-color: #2196F3;
  display: block;
  margin-left: auto;
  margin-right: auto;
}

.next-feedback-btn:hover {
  background-color: #0b7dda;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .feedback-grid {
    grid-template-columns: 1fr;
  }
  
  .small-video, .emotion-graph {
    width: 95%;
  }
  
  .summary-row {
    flex-direction: column;
    gap: 10px;
  }
}
</style>
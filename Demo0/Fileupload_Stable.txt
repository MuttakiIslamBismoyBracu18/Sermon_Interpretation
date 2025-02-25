<template>
  <div>
    <input type="file" @change="onFileChange" ref="fileInput" />
    <button @click="triggerFileUpload" class="btn upload-btn">Upload</button>
  </div>
</template>

<script>
export default {
  methods: {
    onFileChange(event) {
      const file = event.target.files[0];
      if (file) {
        this.$emit("file-uploaded", file);
      } else {
        alert("No file selected. Please choose a video file to upload.");
      }
    },
    triggerFileUpload() {
      this.$refs.fileInput.click();
    },
  },
};
</script>

<style scoped>
input[type="file"] {
  display: none; /* Hide default file input for custom styling */
}

.upload-btn {
  background-color: #b33939;
  color: white;
  padding: 8px 16px;
  border: none;
  border-radius: 5px;
  cursor: pointer;
}

.upload-btn:hover {
  background-color: #ff5e57;
}
</style>

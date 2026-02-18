# LinkedIn Post Draft

---

Back in university, I built an AI that could tell whether a concrete wall was cracking — just from a photo.

The original idea was to attach it to a drone. Fly it alongside a building, let the AI do the inspection, flag the cracks. No scaffolding. No engineer dangling off a rope.

I was a Software Engineering student at TARUMT at the time, working on this alone. I trained a ResNet50 model — a deep learning architecture pre-trained on over a million images — on 33,000 photos of concrete walls, half cracked, half intact. After about 4.5 hours of training on a GTX 1060, the model converged at **98.2% accuracy**.

I demo'd it to my lecturer with a desktop app I built on top of it.

---

Recently I revisited the project and added something I wish I had back then: **Grad-CAM visualization**.

Grad-CAM is a technique that shows you *where* the model is looking when it makes a decision. Instead of just getting a "crack detected" output, you get a heatmap overlaid on the image — the hotspots are where the neural network is paying attention.

It turns a black-box model into something you can actually explain to a client, a regulator, or a lecturer.

The third image is the output: original photo → heatmap → prediction overlaid. The model isn't guessing. It's reading the crack lines the same way a structural engineer would.

---

I'm an iOS engineer now, so this isn't my day job. But I think about this project a lot, because it sits at the intersection of two things I find genuinely exciting:

1. Software that works on *real physical problems* — not just screens
2. AI that you can **see** and **trust**, not just deploy and hope

If you're building anything in structural monitoring, infrastructure inspection, or drone tech — I'd love to hear what you're working on.

---

**Tech:** Python · TensorFlow · ResNet50 · Transfer Learning · Grad-CAM · OpenCV · Streamlit

#MachineLearning #ComputerVision #DeepLearning #TransferLearning #ExplainableAI #StructuralEngineering #Python #SideProject

---

## Notes for posting

- **Attach 3 images in order:**
  1. A screenshot of the GUI showing a detected crack (already in `reports/figures/ui_detected.png`)
  2. The crop-overlay output (already in `reports/figures/detected_whole_image.png`)
  3. The Grad-CAM output — generate it first with:
     ```
     python -m crack_detection.gradcam models/26102021-231847-e100.h5 <your_image.jpg>
     ```
     Output saves to `reports/figures/gradcam_output.png`

- **Best time to post:** Tuesday–Thursday, 8–10am your local time
- **Engagement tip:** Ask a question at the end ("Are you working on anything in infrastructure AI?") — already included above
- **Keep the comments section open** — this kind of post tends to attract civil engineers and ML people who have strong opinions about both

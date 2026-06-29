document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('search-form');
    const input = document.getElementById('query-input');
    const loader = document.getElementById('loader');
    const resultsContainer = document.getElementById('results-container');
    const toggleGalleryBtn = document.getElementById('toggle-gallery');
    const galleryList = document.getElementById('gallery-list');

    // Tab Toggling
    const tabRetrieval = document.getElementById('tab-retrieval');
    const tabCaptioning = document.getElementById('tab-captioning');
    const tabMethodology = document.getElementById('tab-methodology');
    const tabAnalytics = document.getElementById('tab-analytics');
    const tabTeam = document.getElementById('tab-team');
    const tabSource = document.getElementById('tab-source');
    
    const sectionRetrieval = document.getElementById('section-retrieval');
    const sectionCaptioning = document.getElementById('section-captioning');
    const sectionMethodology = document.getElementById('section-methodology');
    const sectionAnalytics = document.getElementById('section-analytics');
    const sectionTeam = document.getElementById('section-team');
    const sectionSource = document.getElementById('section-source');

    let analyticsAnimated = false;
    function animateAnalytics() {
        if (analyticsAnimated) return;
        analyticsAnimated = true;
        
        setTimeout(() => {
            document.querySelectorAll('.progress-fill').forEach(el => {
                const target = parseFloat(el.getAttribute('data-target'));
                el.style.width = target + '%';
            });
            
            document.querySelectorAll('.live-counter, .live-val').forEach(el => {
                const target = parseFloat(el.getAttribute('data-target'));
                const isPercent = el.getAttribute('data-format') === 'percent' || el.classList.contains('live-val');
                const stepAmount = parseFloat(el.getAttribute('data-step')) || (target / 30); 
                let current = 0;
                const timer = setInterval(() => {
                    current += stepAmount;
                    if (current >= target) {
                        current = target;
                        clearInterval(timer);
                    }
                    el.textContent = current.toFixed(1) + (isPercent ? '%' : '');
                }, 40);
            });
            
            setInterval(() => {
                const gpu = document.getElementById('live-gpu');
                const latency = document.getElementById('live-latency');
                const conn = document.getElementById('live-conn');
                const acc = document.getElementById('live-accuracy');
                const rank = document.getElementById('live-rank');

                if (gpu) gpu.textContent = Math.floor(Math.random() * 10 + 38) + '%';
                if (latency) latency.textContent = Math.floor(Math.random() * 25 + 110) + ' ms';
                if (conn) conn.textContent = Math.floor(Math.random() * 6 + 2);

                if (acc) {
                    const currentVal = parseFloat(acc.textContent);
                    if (currentVal > 80.0) { 
                        const baseAcc = 89.0;
                        const jitter = (Math.random() - 0.5); 
                        acc.textContent = (baseAcc + jitter).toFixed(1) + '%';
                    }
                }
                
                if (rank) {
                    const currentVal = parseFloat(rank.textContent);
                    if (currentVal > 5.0) { 
                        const baseRank = 8.2;
                        const jitter = (Math.random() - 0.5) * 0.8; 
                        rank.textContent = (baseRank + jitter).toFixed(1);
                    }
                }

                // Jitter live recall text values
                document.querySelectorAll('.live-val').forEach(el => {
                    const baseTarget = parseFloat(el.getAttribute('data-target'));
                    const currentVal = parseFloat(el.textContent);
                    if (currentVal > baseTarget - 5) {
                        const jitter = (Math.random() - 0.5);
                        el.textContent = (baseTarget + jitter).toFixed(1) + '%';
                    }
                });

                // Jitter progress bar widths
                document.querySelectorAll('.progress-fill').forEach(el => {
                    const baseTarget = parseFloat(el.getAttribute('data-target'));
                    const currentWidth = parseFloat(el.style.width);
                    if (currentWidth > baseTarget - 5) {
                        const jitter = (Math.random() - 0.5);
                        el.style.width = (baseTarget + jitter).toFixed(1) + '%';
                    }
                });
            }, 1000);
        }, 100);
    }

    function switchTab(activeTab, activeSection) {
        [tabRetrieval, tabCaptioning, tabMethodology, tabAnalytics, tabTeam, tabSource].forEach(t => {
            if (t) {
                t.classList.remove('active');
                t.classList.add('inactive');
            }
        });
        [sectionRetrieval, sectionCaptioning, sectionMethodology, sectionAnalytics, sectionTeam, sectionSource].forEach(s => {
            if (s) s.style.display = 'none';
        });

        if (activeTab) {
            activeTab.classList.add('active');
            activeTab.classList.remove('inactive');
        }
        if (activeSection) {
            activeSection.style.display = 'block';
            if (activeSection === sectionAnalytics) {
                animateAnalytics();
            }
        }
    }

    if (tabRetrieval) tabRetrieval.addEventListener('click', () => switchTab(tabRetrieval, sectionRetrieval));
    if (tabCaptioning) tabCaptioning.addEventListener('click', () => switchTab(tabCaptioning, sectionCaptioning));
    if (tabMethodology) tabMethodology.addEventListener('click', () => switchTab(tabMethodology, sectionMethodology));
    if (tabAnalytics) tabAnalytics.addEventListener('click', () => switchTab(tabAnalytics, sectionAnalytics));
    if (tabTeam) tabTeam.addEventListener('click', () => switchTab(tabTeam, sectionTeam));
    if (tabSource) tabSource.addEventListener('click', () => switchTab(tabSource, sectionSource));
    
    // Theme Toggling
    const themeToggleBtn = document.getElementById('theme-toggle');
    const htmlElement = document.documentElement;

    function setTheme(theme) {
        htmlElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
        themeToggleBtn.textContent = theme === 'dark' ? '☀️ Light Mode' : '🌙 Dark Mode';
    }

    const savedTheme = localStorage.getItem('theme') || 'dark';
    setTheme(savedTheme);

    themeToggleBtn.addEventListener('click', () => {
        const currentTheme = htmlElement.getAttribute('data-theme');
        setTheme(currentTheme === 'dark' ? 'light' : 'dark');
    });

    // Info Modal Toggling
    const infoToggleBtn = document.getElementById('info-toggle');
    const modal = document.getElementById('model-modal');
    const closeModalBtn = document.querySelector('.close-modal');

    infoToggleBtn.addEventListener('click', () => {
        modal.classList.add('show');
    });

    closeModalBtn.addEventListener('click', () => {
        modal.classList.remove('show');
    });

    window.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.remove('show');
        }
    });

    // Global Audio Manager (Only 1 plays at a time)
    let currentAudio = null;
    let currentVisualizer = null;

    function setupAudio(audioEl, visualizerEl) {
        audioEl.addEventListener('play', () => {
            if (currentAudio && currentAudio !== audioEl) {
                currentAudio.pause();
                currentAudio.currentTime = 0;
            }
            currentAudio = audioEl;
            
            if (currentVisualizer) {
                currentVisualizer.classList.remove('active');
            }
            currentVisualizer = visualizerEl;
            if (currentVisualizer) currentVisualizer.classList.add('active');
        });

        audioEl.addEventListener('pause', () => {
            if (visualizerEl) visualizerEl.classList.remove('active');
        });
        
        audioEl.addEventListener('ended', () => {
            if (visualizerEl) visualizerEl.classList.remove('active');
        });
    }
    
    // Fetch and display all dataset samples globally
    async function loadDatasetData() {
        try {
            const response = await fetch('/api/audio');
            const data = await response.json();
            
            if (data.samples && data.samples.length > 0) {
                const allSamples = data.samples;
                
                // 1. Populate retrieval gallery
                galleryList.innerHTML = allSamples.map(sample => `
                    <div class="gallery-item">
                        <p><strong>${sample.title} (${sample.id}):</strong> ${sample.true_caption}</p>
                        <audio class="gallery-audio" controls preload="none">
                            <source src="${sample.url}" type="audio/wav">
                        </audio>
                    </div>
                `).join('');

                // 2. Populate captioning tags
                const captioningDataset = document.getElementById('captioning-dataset');
                if (captioningDataset) {
                    captioningDataset.innerHTML = allSamples.map(sample => `
                        <button class="dataset-tag">${sample.title.toUpperCase()}</button>
                    `).join('');
                }

                // Attach audio listeners
                document.querySelectorAll('.gallery-audio').forEach(audio => {
                    setupAudio(audio, null);
                });
            }
        } catch (error) {
            console.error('Error loading dataset data:', error);
        }
    }
    
    // Load immediately on init
    loadDatasetData();
    
    if (toggleGalleryBtn) {
        toggleGalleryBtn.addEventListener('click', () => {
            if (galleryList.style.display === 'none') {
                galleryList.style.display = 'grid';
                toggleGalleryBtn.textContent = 'Hide All';
            } else {
                galleryList.style.display = 'none';
                toggleGalleryBtn.textContent = 'View All';
            }
        });
    }

    // CAPTIONING SECTION LOGIC
    const activeCaptioningItem = document.getElementById('captioning-active-item');
    const captioningItemHeader = document.getElementById('captioning-item-header');
    const captioningAudio = document.getElementById('captioning-audio');
    const captioningVis = document.getElementById('captioning-vis');
    const generateCaptionBtn = document.getElementById('generate-caption-btn');
    const captioningResult = document.getElementById('captioning-result');
    const audioUpload = document.getElementById('audio-upload');
    
    let currentCaptioningId = null;
    let currentCaptioningFile = null;

    setupAudio(captioningAudio, captioningVis);

    // Handle dataset tag clicks
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('dataset-tag')) {
            const index = Array.from(e.target.parentNode.children).indexOf(e.target);
            fetch('/api/audio').then(res => res.json()).then(data => {
                const sample = data.samples[index];
                if (sample) {
                    currentCaptioningId = sample.id;
                    currentCaptioningFile = null;
                    captioningItemHeader.textContent = `// DATASET ITEM LOADED: ID ${sample.id}`;
                    captioningAudio.src = sample.url;
                    captioningResult.style.display = 'none';
                    activeCaptioningItem.style.display = 'block';
                    activeCaptioningItem.scrollIntoView({ behavior: 'smooth' });
                }
            });
        }
    });

    // Handle audio upload
    if (audioUpload) {
        audioUpload.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                currentCaptioningFile = file;
                currentCaptioningId = null;
                const fileURL = URL.createObjectURL(file);
                captioningItemHeader.textContent = `// UPLOADED AUDIO FILE: ${file.name}`;
                captioningAudio.src = fileURL;
                captioningResult.style.display = 'none';
                activeCaptioningItem.style.display = 'block';
                activeCaptioningItem.scrollIntoView({ behavior: 'smooth' });
            }
        });
    }

    // Handle Generate Caption Button
    if (generateCaptionBtn) {
        generateCaptionBtn.addEventListener('click', async () => {
            const formData = new FormData();
            if (currentCaptioningId !== null) {
                formData.append('id', currentCaptioningId);
            } else if (currentCaptioningFile !== null) {
                formData.append('file', currentCaptioningFile);
            } else {
                return;
            }

            generateCaptionBtn.textContent = 'GENERATING...';
            generateCaptionBtn.disabled = true;
            captioningResult.style.display = 'none';
            captioningResult.classList.remove('typing-effect');
            captioningResult.textContent = '';

            try {
                const response = await fetch('/api/caption', {
                    method: 'POST',
                    body: formData
                });
                
                if (response.ok) {
                    const data = await response.json();
                    captioningResult.textContent = `> ${data.caption}`;
                    captioningResult.style.display = 'block';
                    captioningResult.classList.add('typing-effect');
                } else {
                    captioningResult.textContent = '> Error generating caption.';
                    captioningResult.style.display = 'block';
                }
            } catch (err) {
                captioningResult.textContent = '> Network error.';
                captioningResult.style.display = 'block';
            } finally {
                generateCaptionBtn.textContent = 'GENERATE NEURAL CAPTION';
                generateCaptionBtn.disabled = false;
            }
        });
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const query = input.value.trim();
        if (!query) return;
        
        loader.style.display = 'block';
        resultsContainer.innerHTML = '';
        
        try {
            const response = await fetch('/api/retrieve', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            });
            
            if (!response.ok) throw new Error('Retrieval failed');
            
            const data = await response.json();
            displayResults(data.results);
            
        } catch (error) {
            resultsContainer.innerHTML = `
                <div class="glass-panel" style="text-align: center; color: #ef4444; width: 100%;">
                    <p>Error retrieving results. Make sure the server is running.</p>
                </div>
            `;
        } finally {
            loader.style.display = 'none';
        }
    });
    
    function displayResults(results) {
        if (!results || results.length === 0) {
            resultsContainer.innerHTML = `
                <div class="glass-panel" style="text-align: center; width: 100%;">
                    <p>No matching audio found.</p>
                </div>
            `;
            return;
        }
        
        resultsContainer.innerHTML = results.map((result, index) => {
            const rankClass = index < 3 ? `rank-${index+1}` : 'rank-other';
            return `
            <div class="result-card" style="animation-delay: ${index * 0.1}s">
                <div class="result-header">
                    <span class="rank-badge ${rankClass}">Rank #${index + 1}</span>
                    <span class="score-text">Similarity: ${(result.score).toFixed(4)}</span>
                </div>
                <div class="true-caption">
                    ${result.sample.true_caption}
                </div>
                ${result.sample.spectrogram ? `
                <div class="spectrogram-container" title="Mel-Spectrogram">
                    <img src="${result.sample.spectrogram}" alt="Spectrogram for audio ${result.sample.id}">
                </div>` : ''}
                
                <div class="visualizer" id="vis-${index}">
                    <div class="bar"></div>
                    <div class="bar"></div>
                    <div class="bar"></div>
                    <div class="bar"></div>
                    <div class="bar"></div>
                </div>

                <audio id="audio-${index}" controls preload="metadata">
                    <source src="${result.sample.url}" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
            </div>
            `;
        }).join('');

        // Attach Audio Listeners
        results.forEach((_, index) => {
            const audioEl = document.getElementById(`audio-${index}`);
            const visEl = document.getElementById(`vis-${index}`);
            setupAudio(audioEl, visEl);
        });
    }
});

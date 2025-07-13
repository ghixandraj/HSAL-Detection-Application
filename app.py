/* --- Global & Font --- */
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif !important;
    background-color: #0E1117 !important;
    color: #e0e0e0 !important;
    position: relative;
    overflow-x: hidden;
}

.stApp {
    background-color: #0E1117 !important;
    position: relative;
    z-index: 1; /* Tambahkan z-index untuk konten utama */
}

/* Animated texture background */
body::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: 
        radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.1) 0%, transparent 50%),
        radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.08) 0%, transparent 50%),
        radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.06) 0%, transparent 50%);
    animation: textureMove 20s ease-in-out infinite;
    pointer-events: none;
    z-index: -2; /* Lebih rendah dari konten */
}

/* Floating particles effect */
body::after {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: 
        radial-gradient(2px 2px at 20px 30px, rgba(255, 255, 255, 0.05), transparent),
        radial-gradient(2px 2px at 40px 70px, rgba(255, 255, 255, 0.03), transparent),
        radial-gradient(1px 1px at 90px 40px, rgba(255, 255, 255, 0.04), transparent),
        radial-gradient(1px 1px at 130px 80px, rgba(255, 255, 255, 0.02), transparent),
        radial-gradient(2px 2px at 160px 30px, rgba(255, 255, 255, 0.03), transparent);
    background-repeat: repeat;
    background-size: 200px 100px;
    animation: particlesFloat 15s linear infinite;
    pointer-events: none;
    z-index: -1; /* Lebih rendah dari konten */
}

/* Gradient mesh overlay - PINDAHKAN KE BODY */
body::after {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: 
        linear-gradient(45deg, rgba(30, 41, 59, 0.1) 0%, transparent 50%),
        linear-gradient(-45deg, rgba(15, 23, 42, 0.1) 0%, transparent 50%),
        linear-gradient(90deg, rgba(51, 65, 85, 0.05) 0%, transparent 50%),
        /* Gabungkan dengan particles */
        radial-gradient(2px 2px at 20px 30px, rgba(255, 255, 255, 0.05), transparent),
        radial-gradient(2px 2px at 40px 70px, rgba(255, 255, 255, 0.03), transparent),
        radial-gradient(1px 1px at 90px 40px, rgba(255, 255, 255, 0.04), transparent),
        radial-gradient(1px 1px at 130px 80px, rgba(255, 255, 255, 0.02), transparent),
        radial-gradient(2px 2px at 160px 30px, rgba(255, 255, 255, 0.03), transparent);
    background-repeat: repeat;
    background-size: 200px 100px;
    animation: combinedAnimation 20s ease-in-out infinite;
    pointer-events: none;
    z-index: -1;
}

/* Keyframe animations */
@keyframes textureMove {
    0%, 100% {
        transform: translateX(0) translateY(0) rotate(0deg);
    }
    25% {
        transform: translateX(-10px) translateY(-20px) rotate(1deg);
    }
    50% {
        transform: translateX(10px) translateY(10px) rotate(-1deg);
    }
    75% {
        transform: translateX(-5px) translateY(-10px) rotate(0.5deg);
    }
}

@keyframes particlesFloat {
    0% {
        transform: translateY(0) translateX(0);
    }
    50% {
        transform: translateY(-20px) translateX(10px);
    }
    100% {
        transform: translateY(0) translateX(0);
    }
}

@keyframes combinedAnimation {
    0%, 100% {
        transform: translateX(0) translateY(0) scale(1);
    }
    33% {
        transform: translateX(5px) translateY(-10px) scale(1.02);
    }
    66% {
        transform: translateX(-5px) translateY(5px) scale(0.98);
    }
}

/* Pastikan semua konten Streamlit memiliki z-index yang tepat */
.stApp > div,
.main .block-container,
.stSidebar,
.stSidebar > div,
.main-container,
.header-box,
.feature-section,
.info-card,
/* Tambahkan selector Streamlit yang spesifik */
[data-testid="stSidebar"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
.stButton,
.stTextInput,
.stTextArea,
.stSelectbox,
.stVideo,
.stProgress,
.stAlert,
.stSuccess,
.stError,
.stWarning,
.stInfo,
.stExpander,
.stSpinner,
.stMarkdown,
.stSubheader,
.stHeader {
    position: relative;
    z-index: 10 !important; /* Pastikan semua konten di atas background */
}

/* --- Main Container & Layout --- */
.main-container {
    max-width: 1200px;
    margin: auto;
    padding: 2rem 1rem;
    position: relative;
    z-index: 10;
}

/* --- STYLES HEADER BARU --- */
.header-box {
    display: flex;
    padding: 25px;
    flex-direction: column;
    align-items: center;
    gap: 20px;
    align-self: stretch;
    border-radius: 25px;
    border: none;
    background: rgba(0, 0, 0, 0.20);
    box-shadow: 0px 4px 4px 0px rgba(0, 0, 0, 0.50);
    margin-bottom: 4rem;
    position: relative;
    z-index: 15;
}

.top-pill {
    display: flex;
    padding: 10px 20px;
    justify-content: center;
    align-items: center;
    gap: 10px;
    border-radius: 100px;
    border: none;
    background: rgba(255, 255, 255, 0.05);
    box-shadow: 0px 4px 4px 0px rgba(0, 0, 0, 0.50);
    backdrop-filter: blur(5px);
    color: #FFF;
    font-family: 'Poppins', sans-serif;
    font-size: 18px;
    font-weight: 400;
    margin: 0;
    position: relative;
    z-index: 16;
    text-align: center;
}

.main-title {
    width: 100%;
    max-width: 680px;
    text-align: center;
    font-family: 'Poppins', sans-serif;
    font-size: 36px;
    font-style: normal;
    font-weight: 900;
    line-height: normal;
    background: linear-gradient(90deg, #FFF 0%, #39EDEA 50%, #686ACF 100%);
    background-clip: text;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 auto;
    position: relative;
    z-index: 16;
}

.subtitle {
    align-self: stretch;
    color: #FFF;
    text-align: center;
    font-family: 'Poppins', sans-serif;
    font-size: 18px !important;
    font-style: normal;
    font-weight: 200;
    line-height: normal;
    margin: 0;
    max-width: none;
    position: relative;
    z-index: 16;
}

/* --- Feature Section --- */
.feature-section {
    display: flex;
    padding: 25px;
    flex-direction: column;
    align-items: center;
    gap: 25px;
    align-self: stretch;
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    background: linear-gradient(135deg, 
        rgba(128, 128, 128, 0.3) 0%,
        rgba(64, 64, 64, 0.4) 50%,
        rgba(0, 0, 0, 0.5) 100%
    );
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    box-shadow: 
        0px 8px 32px rgba(0, 0, 0, 0.37),
        inset 0px 1px 0px rgba(255, 255, 255, 0.1);
    margin-bottom: 3rem;
    overflow: visible;
    position: relative;
    width: 100%;
    z-index: 15;
}

/* Tambahan untuk efek glass yang lebih dramatis */
.feature-section::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, 
        rgba(255, 255, 255, 0.05) 0%,
        rgba(255, 255, 255, 0.02) 50%,
        rgba(255, 255, 255, 0.0) 100%
    );
    border-radius: 15px;
    pointer-events: none;
    z-index: -1; /* Relative terhadap parent */
}

/* Hover effect untuk interaktivitas */
.feature-section:hover {
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 
        0px 12px 40px rgba(0, 0, 0, 0.45),
        inset 0px 1px 0px rgba(255, 255, 255, 0.15);
    transform: translateY(-2px);
    transition: all 0.3s ease;
}

.feature-section > * {
    position: relative;
    z-index: 2;
}

.section-title-pill {
    display: flex;
    padding: 10px 20px;
    justify-content: center;
    align-items: center;
    gap: 10px;
    border-radius: 100px;
    border: none;
    background: rgba(255, 255, 255, 0.05);
    box-shadow: 0px 4px 4px 0px rgba(0, 0, 0, 0.50);
    backdrop-filter: blur(5px);
    color: #FFF;
    font-weight: 500;
    margin: 0;
    z-index: 16;
    position: relative;
}

.category-box {
    text-align: center;
    margin: 0;
    position: relative;
    z-index: 16;
}

.category-title {
    align-self: stretch;
    text-align: center;
    font-size: 40px !important;
    font-weight: 800;
    line-height: normal;
    background: linear-gradient(270deg, #7B39C6 46.48%, #39EDEA 74.27%, #FFF 99.99%);
    background-clip: text;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
    position: relative;
    z-index: 16;
}

.category-text {
    align-self: stretch;
    color: #FFF;
    text-align: center;
    font-size: 20px !important;
    font-weight: 500;
    line-height: normal;
    max-width: 750px;
    margin: 0 auto;
    position: relative;
    z-index: 16;
}

.category-text span {
    font-weight: 700;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.category-text .highlight-1 { 
    background: linear-gradient(90deg, #39EDEA 31.26%, #686ACF 46.26%); 
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
}

.category-text .highlight-2 { 
    background: linear-gradient(90deg, #39EDEA 58.12%, #686ACF 67.66%); 
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
}

.category-text .highlight-3 { 
    background: linear-gradient(90deg, #39EDEA 69.20%, #686ACF 79.62%); 
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
}

.category-text .highlight-4 { 
    background: linear-gradient(90deg, #39EDEA 84.04%, #686ACF 97.36%); 
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
}

/* --- INFO CARDS --- */
.cards-container {
    display: flex;
    justify-content: center;
    gap: 2rem;
    flex-wrap: wrap;
    position: relative;
    z-index: 16;
    width: 100%;
    margin: 0 auto;
}

.info-card {
    display: flex;
    padding: 25px;
    flex-direction: column;
    align-items: flex-start;
    gap: 5px;
    flex: 1;
    align-self: stretch;
    border-radius: 15px;
    background: linear-gradient(180deg, #192221 31.25%, #32224D 65.62%, #4A2179 99.99%);
    box-shadow: 0px 4px 5.7px 0px rgba(0, 0, 0, 0.50);
    text-align: left;
    transition: all 0.2s ease-in-out;
    position: relative;
    z-index: 16;
}

.info-card:hover {
    transform: scale(1.03);
    box-shadow: 0px 6px 10px 0px rgba(0, 0, 0, 0.60);
}

.card-icon {
    display: flex;
    width: 60px;
    height: 60px;
    justify-content: center;
    align-items: center;
    border-radius: 100px;
    background: linear-gradient(180deg, #414141 0%, #212121 100%);
    margin: 0;
    position: relative;
    z-index: 17;
}

.card-icon svg {
    width: 36px;
    height: 36px;
    flex-shrink: 0;
    fill: #39EDEA;
}

.card-title,
.card-subtitle {
    display: block;
    margin: 0;
    padding: 0;
    position: relative;
    z-index: 17;
}

.card-title {
    font-size: 1.5rem;
    font-weight: 900;
    color: #ffffff;
    line-height: 1.1;
}

.card-subtitle {
    font-size: 0.9rem;
    color: #cccccc;
    font-weight: 700;
    line-height: 1.2;
}

.card-content {
    font-size: 1rem;
    color: #e0e0e0;
    line-height: 1.5;
    font-weight: 500;
    position: relative;
    z-index: 17;
}

/* --- Responsive Design --- */
@media (max-width: 768px) {
    .main-title {
        font-size: 28px;
    }
    .subtitle {
        font-size: 16px;
    }
    .category-title {
        font-size: 28px;
    }
    .cards-container {
        flex-direction: column;
        align-items: stretch; 
        gap: 1rem;
    }
    .info-card {
        flex: 1;
        width: 100%;
    }
}
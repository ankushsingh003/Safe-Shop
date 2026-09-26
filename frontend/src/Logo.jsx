import React from "react";

export default function SafeShopLogo({ size = 32, showText = true, textVariant = "default" }) {
  return (
    <div style={{ display: "inline-flex", alignItems: "center", gap: 10, userSelect: "none" }}>
      {/* Sleek Geometric Shield & Neural Nexus Vector Logo */}
      <svg
        width={size}
        height={size}
        viewBox="0 0 36 36"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        style={{ flexShrink: 0, filter: "drop-shadow(0 2px 8px rgba(0, 0, 0, 0.6))" }}
      >
        {/* Outer Shield Hex Frame with Metallic Gradient */}
        <path
          d="M18 2.5L5 7.5V17.5C5 25.2 10.6 31.4 18 33.5C25.4 31.4 31 25.2 31 17.5V7.5L18 2.5Z"
          fill="url(#outerGrad)"
          stroke="url(#outerStroke)"
          strokeWidth="1.5"
        />
        {/* Inner Shield Inset */}
        <path
          d="M18 6.5L8.5 10.5V17.8C8.5 23.5 12.6 28.2 18 29.8C23.4 28.2 27.5 23.5 27.5 17.8V10.5L18 6.5Z"
          fill="url(#innerGrad)"
          stroke="rgba(255, 255, 255, 0.08)"
          strokeWidth="1"
        />
        {/* Precision Core Architecture (Security Node & Lines) */}
        <path
          d="M18 10V26"
          stroke="url(#lineGrad)"
          strokeWidth="1.2"
          strokeLinecap="round"
          strokeDasharray="2 2"
        />
        <path
          d="M11.5 18H24.5"
          stroke="url(#lineGrad)"
          strokeWidth="1.2"
          strokeLinecap="round"
        />
        {/* Center Node */}
        <circle cx="18" cy="18" r="3.2" fill="#FFFFFF" />
        <circle cx="18" cy="18" r="5" stroke="rgba(255, 255, 255, 0.4)" strokeWidth="1" />
        {/* Diagonal Sentinel Accents */}
        <circle cx="13" cy="13" r="1.4" fill="#A1A1AA" />
        <circle cx="23" cy="13" r="1.4" fill="#A1A1AA" />
        <circle cx="18" cy="24" r="1.4" fill="#A1A1AA" />

        <defs>
          <linearGradient id="outerGrad" x1="18" y1="2.5" x2="18" y2="33.5" gradientUnits="userSpaceOnUse">
            <stop stopColor="#1C1D24" />
            <stop offset="1" stopColor="#0B0C0E" />
          </linearGradient>
          <linearGradient id="outerStroke" x1="5" y1="2.5" x2="31" y2="33.5" gradientUnits="userSpaceOnUse">
            <stop stopColor="#FFFFFF" stopOpacity="0.8" />
            <stop offset="0.4" stopColor="#71717A" />
            <stop offset="1" stopColor="#27272A" />
          </linearGradient>
          <linearGradient id="innerGrad" x1="18" y1="6.5" x2="18" y2="29.8" gradientUnits="userSpaceOnUse">
            <stop stopColor="#14151B" stopOpacity="0.9" />
            <stop offset="1" stopColor="#08090C" />
          </linearGradient>
          <linearGradient id="lineGrad" x1="11.5" y1="18" x2="24.5" y2="18" gradientUnits="userSpaceOnUse">
            <stop stopColor="#71717A" />
            <stop offset="0.5" stopColor="#FFFFFF" />
            <stop offset="1" stopColor="#71717A" />
          </linearGradient>
        </defs>
      </svg>

      {showText && (
        <div style={{ display: "flex", flexDirection: "column", lineHeight: 1.1 }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 3 }}>
            <span
              style={{
                fontSize: size >= 32 ? 20 : 15,
                fontWeight: 700,
                letterSpacing: "-0.03em",
                color: "#FFFFFF",
                fontFamily: "var(--font-sans, 'Plus Jakarta Sans', sans-serif)",
              }}
            >
              SafeShop
            </span>
            <span
              style={{
                fontSize: size >= 32 ? 14 : 11,
                fontWeight: 600,
                letterSpacing: "-0.01em",
                color: "#71717A",
                fontFamily: "var(--font-sans, 'Plus Jakarta Sans', sans-serif)",
              }}
            >
              .ai
            </span>
          </div>
          {textVariant === "full" && (
            <span
              style={{
                fontSize: 9,
                fontWeight: 600,
                letterSpacing: "0.1em",
                color: "#52525B",
                textTransform: "uppercase",
                marginTop: 2,
                fontFamily: "var(--font-mono)",
              }}
            >
              Security Operations
            </span>
          )}
        </div>
      )}
    </div>
  );
}

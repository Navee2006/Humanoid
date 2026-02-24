import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

const Visualizer = ({ audioData, isListening, intensity = 0, width = 600, height = 400 }) => {
    const canvasRef = useRef(null);
    const audioDataRef = useRef(audioData);
    const intensityRef = useRef(intensity);
    const isListeningRef = useRef(isListening);
    const timeRef = useRef(0);

    useEffect(() => {
        audioDataRef.current = audioData;
        intensityRef.current = intensity;
        isListeningRef.current = isListening;
    }, [audioData, intensity, isListening]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        canvas.width = width;
        canvas.height = height;

        const ctx = canvas.getContext('2d');
        let animationId;

        const draw = () => {
            const w = canvas.width;
            const h = canvas.height;
            const centerX = w / 2;
            const centerY = h / 2;

            const currentIntensity = intensityRef.current;
            const currentIsListening = isListeningRef.current;
            const currentAudioData = audioDataRef.current || [];

            timeRef.current += 0.015;
            const time = timeRef.current;

            // DARK background
            ctx.fillStyle = '#0a0a0a';
            ctx.fillRect(0, 0, w, h);

            const baseRadius = Math.min(w, h) * 0.25;

            // Outer Rings - PURPLE
            for (let i = 0; i < 4; i++) {
                const ringRadius = baseRadius + (i * 35) + (Math.sin(time + i * 0.5) * 3);
                const alpha = 0.4 - (i * 0.08);

                ctx.beginPath();
                ctx.arc(centerX, centerY, ringRadius, 0, Math.PI * 2);
                ctx.strokeStyle = `rgba(168, 85, 247, ${alpha})`;
                ctx.lineWidth = 2;
                ctx.shadowBlur = 15;
                ctx.shadowColor = 'rgba(168, 85, 247, 0.5)';
                ctx.stroke();
                ctx.shadowBlur = 0;
            }

            // Main Circle with PURPLE Gradient
            const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, baseRadius);
            gradient.addColorStop(0, 'rgba(168, 85, 247, 0.5)');
            gradient.addColorStop(0.5, 'rgba(147, 51, 234, 0.3)');
            gradient.addColorStop(1, 'rgba(126, 34, 206, 0.1)');

            ctx.beginPath();
            ctx.arc(centerX, centerY, baseRadius, 0, Math.PI * 2);
            ctx.fillStyle = gradient;
            ctx.fill();

            // WAVEFORM BARS AROUND THE RING
            const barCount = 128;
            const angleStep = (Math.PI * 2) / barCount;

            for (let i = 0; i < barCount; i++) {
                const value = currentIsListening && currentAudioData[i]
                    ? currentAudioData[i] / 255
                    : Math.sin(time * 2 + i * 0.1) * 0.1 + 0.1;

                const angle = i * angleStep - Math.PI / 2;

                // Bars AROUND the ring (outside)
                const startRadius = baseRadius + 45;
                const barHeight = currentIsListening ? value * 60 + 8 : value * 20 + 5;
                const endRadius = startRadius + barHeight;

                const startX = centerX + Math.cos(angle) * startRadius;
                const startY = centerY + Math.sin(angle) * startRadius;
                const endX = centerX + Math.cos(angle) * endRadius;
                const endY = centerY + Math.sin(angle) * endRadius;

                // Purple gradient for bars
                const barGradient = ctx.createLinearGradient(startX, startY, endX, endY);
                barGradient.addColorStop(0, 'rgba(168, 85, 247, 0.9)');
                barGradient.addColorStop(0.5, 'rgba(217, 70, 239, 0.8)');
                barGradient.addColorStop(1, 'rgba(168, 85, 247, 0.4)');

                ctx.beginPath();
                ctx.moveTo(startX, startY);
                ctx.lineTo(endX, endY);
                ctx.strokeStyle = barGradient;
                ctx.lineWidth = Math.max(2, (Math.PI * 2 * startRadius) / barCount - 1);
                ctx.lineCap = 'round';
                ctx.shadowBlur = 8;
                ctx.shadowColor = 'rgba(168, 85, 247, 0.6)';
                ctx.stroke();
                ctx.shadowBlur = 0;
            }

            // Pulsing Core Circle - BRIGHT PURPLE
            const pulseRadius = 25 + (currentIntensity * 20) + (Math.sin(time * 3) * 4);
            const coreGradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, pulseRadius);
            coreGradient.addColorStop(0, 'rgba(217, 70, 239, 1)');
            coreGradient.addColorStop(0.4, 'rgba(168, 85, 247, 0.9)');
            coreGradient.addColorStop(1, 'rgba(147, 51, 234, 0.3)');

            ctx.beginPath();
            ctx.arc(centerX, centerY, pulseRadius, 0, Math.PI * 2);
            ctx.fillStyle = coreGradient;
            ctx.shadowBlur = 40;
            ctx.shadowColor = '#a855f7';
            ctx.fill();
            ctx.shadowBlur = 0;

            // Rotating Accent Arcs - PURPLE
            const rotationSpeed = currentIsListening ? 0.3 : 0.15;
            const rotation = time * rotationSpeed;

            for (let i = 0; i < 6; i++) {
                const angle = rotation + (i * Math.PI / 3);
                const arcRadius = baseRadius + 30;
                const arcLength = Math.PI / 8;

                ctx.beginPath();
                ctx.arc(centerX, centerY, arcRadius, angle, angle + arcLength);
                ctx.strokeStyle = 'rgba(168, 85, 247, 0.6)';
                ctx.lineWidth = 2;
                ctx.shadowBlur = 10;
                ctx.shadowColor = 'rgba(168, 85, 247, 0.8)';
                ctx.stroke();
                ctx.shadowBlur = 0;
            }

            // Particle Effects - PURPLE
            if (currentIsListening) {
                for (let i = 0; i < 16; i++) {
                    const particleAngle = (i / 16) * Math.PI * 2 + time * 0.5;
                    const particleRadius = baseRadius + 38 + Math.sin(time * 3 + i) * 8;
                    const px = centerX + Math.cos(particleAngle) * particleRadius;
                    const py = centerY + Math.sin(particleAngle) * particleRadius;

                    ctx.beginPath();
                    ctx.arc(px, py, 2, 0, Math.PI * 2);
                    ctx.fillStyle = 'rgba(217, 70, 239, 0.9)';
                    ctx.shadowBlur = 12;
                    ctx.shadowColor = '#d946ef';
                    ctx.fill();
                    ctx.shadowBlur = 0;
                }
            }

            animationId = requestAnimationFrame(draw);
        };

        draw();
        return () => cancelAnimationFrame(animationId);
    }, [width, height]);

    return (
        <div className="relative bg-black" style={{ width, height }}>
            {/* HUD Top Label - PURPLE */}
            <motion.div
                className="absolute top-2 left-1/2 transform -translate-x-1/2 z-10 pointer-events-none"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, ease: "easeOut" }}
            >
                <div className="text-purple-400 text-sm font-bold tracking-widest whitespace-nowrap">
                    PROJECT: TEMP • AI SYSTEM
                </div>
            </motion.div>

            {/* Central Logo/Text - PURPLE */}
            <div className="absolute inset-0 flex items-center justify-center z-10 pointer-events-none">
                <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{
                        scale: isListening ? [1, 1.08, 1] : 1,
                        opacity: 1
                    }}
                    transition={{
                        scale: { duration: 2, repeat: Infinity, ease: "easeInOut" },
                        opacity: { duration: 0.6 }
                    }}
                    className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 via-fuchsia-400 to-purple-400 font-black tracking-[0.35em]"
                    style={{
                        fontSize: Math.min(width, height) * 0.13,
                        fontFamily: "'Orbitron', 'Rajdhani', sans-serif",
                        filter: 'drop-shadow(0 0 30px rgba(168,85,247,0.9)) drop-shadow(0 0 60px rgba(168,85,247,0.5))'
                    }}
                >
                    Blazer
                </motion.div>
            </div>

            {/* Status Indicator - PURPLE */}
            <motion.div
                className="absolute bottom-4 left-1/2 transform -translate-x-1/2 z-10 pointer-events-none"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.2, ease: "easeOut" }}
            >
                <div className="flex items-center gap-2">
                    <motion.div
                        className={`w-2 h-2 rounded-full ${isListening ? 'bg-fuchsia-400' : 'bg-purple-500'}`}
                        animate={isListening ? {
                            scale: [1, 1.3, 1],
                            opacity: [1, 0.7, 1]
                        } : {}}
                        transition={{ duration: 1.5, repeat: Infinity }}
                        style={{
                            boxShadow: isListening
                                ? '0 0 15px rgba(217,70,239,0.9), 0 0 30px rgba(217,70,239,0.5)'
                                : '0 0 8px rgba(168,85,247,0.6)'
                        }}
                    />
                    <div className="text-purple-400 text-xs font-mono tracking-wide font-semibold">
                        {isListening ? 'ACTIVE' : 'STANDBY'}
                    </div>
                </div>
            </motion.div>

            <canvas
                ref={canvasRef}
                style={{ width: '100%', height: '100%' }}
                className="bg-black"
            />
        </div>
    );
};

export default Visualizer;

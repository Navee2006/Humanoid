import React from 'react';
import { motion } from 'framer-motion';
import { Sun, Moon } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';

const ThemeToggle = () => {
    const { theme, toggleTheme, isDark } = useTheme();

    return (
        <motion.button
            onClick={toggleTheme}
            className={`
                relative w-14 h-7 rounded-full p-1 transition-colors duration-300
                ${isDark ? 'bg-purple-900/50' : 'bg-purple-200'}
                border ${isDark ? 'border-purple-500/30' : 'border-purple-400/50'}
            `}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            title={`Switch to ${isDark ? 'light' : 'dark'} mode`}
        >
            {/* Toggle Circle */}
            <motion.div
                className={`
                    w-5 h-5 rounded-full flex items-center justify-center
                    ${isDark ? 'bg-purple-500' : 'bg-purple-600'}
                    shadow-lg
                `}
                animate={{
                    x: isDark ? 0 : 24
                }}
                transition={{ type: 'spring', stiffness: 500, damping: 30 }}
            >
                {isDark ? (
                    <Moon size={12} className="text-white" />
                ) : (
                    <Sun size={12} className="text-white" />
                )}
            </motion.div>

            {/* Background Icons */}
            <div className="absolute inset-0 flex items-center justify-between px-2 pointer-events-none">
                <Sun
                    size={12}
                    className={`transition-opacity ${isDark ? 'opacity-30' : 'opacity-0'}`}
                    style={{ color: isDark ? '#a855f7' : '#7c3aed' }}
                />
                <Moon
                    size={12}
                    className={`transition-opacity ${isDark ? 'opacity-0' : 'opacity-30'}`}
                    style={{ color: isDark ? '#a855f7' : '#7c3aed' }}
                />
            </div>
        </motion.button>
    );
};

export default ThemeToggle;

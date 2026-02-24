import React, { createContext, useContext, useState, useEffect } from 'react';

const ThemeContext = createContext();

export const useTheme = () => {
    const context = useContext(ThemeContext);
    if (!context) {
        throw new Error('useTheme must be used within ThemeProvider');
    }
    return context;
};

export const ThemeProvider = ({ children }) => {
    const [theme, setTheme] = useState(() => {
        // Load theme from localStorage or default to 'dark'
        return localStorage.getItem('Blazer-theme') || 'dark';
    });

    useEffect(() => {
        // Save theme to localStorage
        localStorage.setItem('Blazer-theme', theme);

        // Apply theme class to document
        document.documentElement.classList.remove('light', 'dark');
        document.documentElement.classList.add(theme);
    }, [theme]);

    const toggleTheme = () => {
        setTheme(prev => prev === 'dark' ? 'light' : 'dark');
    };

    const colors = {
        dark: {
            // Purple theme for dark mode
            primary: '#a855f7',      // purple-500
            primaryLight: '#c084fc', // purple-400
            primaryDark: '#7c3aed',  // purple-600
            accent: '#d946ef',       // fuchsia-500
            background: '#0a0a0a',   // near black
            surface: '#1a1a1a',      // dark gray
            text: '#f3f4f6',         // gray-100
            textSecondary: '#9ca3af',// gray-400
            border: 'rgba(168, 85, 247, 0.2)', // purple with opacity
            glow: 'rgba(168, 85, 247, 0.6)',   // purple glow
        },
        light: {
            // Purple theme for light mode
            primary: '#7c3aed',      // purple-600
            primaryLight: '#a855f7', // purple-500
            primaryDark: '#6d28d9',  // purple-700
            accent: '#c026d3',       // fuchsia-600
            background: '#ffffff',   // white
            surface: '#f9fafb',      // gray-50
            text: '#111827',         // gray-900
            textSecondary: '#6b7280',// gray-500
            border: 'rgba(124, 58, 237, 0.2)', // purple with opacity
            glow: 'rgba(124, 58, 237, 0.4)',   // purple glow (lighter for light mode)
        }
    };

    const value = {
        theme,
        toggleTheme,
        isDark: theme === 'dark',
        colors: colors[theme]
    };

    return (
        <ThemeContext.Provider value={value}>
            {children}
        </ThemeContext.Provider>
    );
};

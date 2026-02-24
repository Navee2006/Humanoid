import React from 'react';
import backgroundImg from '../assets/dreamy-background.jpg';

const DreamyBackground = () => {
    return (
        <div
            className="fixed inset-0 -z-10 bg-cover bg-center bg-no-repeat"
            style={{
                backgroundImage: `url(${backgroundImg})`,
                width: '100vw',
                height: '100vh'
            }}
        />
    );
};

export default DreamyBackground;

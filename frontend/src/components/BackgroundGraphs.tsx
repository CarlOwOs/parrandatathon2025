import React from 'react';
import { motion } from 'framer-motion';

const BackgroundGraphs: React.FC = () => {
  // Create a grid of graphs with more predictable positioning
  const graphs = Array(4).fill(null).map((_, i) => ({
    id: i,
    // Position graphs in a more structured way
    initialX: (i % 2) * 50,  // Creates two columns
    initialY: Math.floor(i / 2) * 50,  // Creates two rows
  }));

  return (
    <div className="absolute inset-0 overflow-hidden opacity-5">
      {graphs.map((graph) => (
        <motion.div
          key={graph.id}
          className="absolute"
          style={{
            width: '800px',
            height: '800px',
            left: `${graph.initialX}%`,
            top: `${graph.initialY}%`,
            transform: `rotate(${graph.id * 45}deg)`, // Rotate each graph differently
          }}
          initial={{ x: 0, y: 0 }}
          animate={{
            x: [0, 40, 0],
            y: [0, 20, 0],
          }}
          transition={{
            duration: 15,
            repeat: Infinity,
            ease: "easeInOut",
            delay: graph.id * 1.5,
          }}
        >
          <svg
            viewBox="0 0 100 100"
            className="w-full h-full"
          >
            {/* Create smoother wave patterns */}
            <motion.path
              d={`
                M 0,50 
                C 20,45 30,55 50,50 
                C 70,45 80,55 100,50
              `}
              stroke="currentColor"
              fill="none"
              strokeWidth="0.3"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{
                duration: 2,
                ease: "easeInOut",
                repeat: Infinity,
                repeatType: "reverse"
              }}
            />
            <motion.path
              d={`
                M 0,60 
                C 25,55 35,65 50,60 
                C 65,55 75,65 100,60
              `}
              stroke="currentColor"
              fill="none"
              strokeWidth="0.3"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{
                duration: 2.5,
                ease: "easeInOut",
                repeat: Infinity,
                repeatType: "reverse",
                delay: 0.5
              }}
            />
          </svg>
        </motion.div>
      ))}
    </div>
  );
};

export default BackgroundGraphs; 
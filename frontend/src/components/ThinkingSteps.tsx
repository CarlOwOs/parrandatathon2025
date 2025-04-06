import React from 'react';
import { motion } from 'framer-motion';

interface ThinkingStepsProps {
  currentStep: number;
  isComplete: boolean;
  response?: string;
}

const steps = [
  {
    title: "Initializing RAG pipeline...",
    description: "Setting up the retrieval system"
  },
  {
    title: "Retrieving relevant documents...",
    description: "Searching through the knowledge base"
  },
  {
    title: "Generating response...",
    description: "Analyzing and composing the answer"
  }
];

const ThinkingSteps: React.FC<ThinkingStepsProps> = ({ currentStep, isComplete, response }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="mt-6 bg-secondary/50 rounded-lg p-4"
    >
      <div className="space-y-4">
        {steps.map((step, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.2 }}
            className={`flex items-start space-x-3 ${
              index < currentStep ? 'text-accent' : 
              index === currentStep ? 'text-white' : 
              'text-gray-500'
            }`}
          >
            <div className="flex-shrink-0 mt-1">
              {index < currentStep ? (
                "✓"
              ) : index === currentStep ? (
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                  className="w-4 h-4 border-2 border-current border-t-transparent rounded-full"
                />
              ) : (
                "○"
              )}
            </div>
            <div>
              <p className="font-medium">{step.title}</p>
              <p className="text-sm opacity-75">{step.description}</p>
            </div>
          </motion.div>
        ))}
      </div>
      
      {isComplete && response && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="mt-6 p-4 bg-secondary rounded-lg"
        >
          <h3 className="text-accent font-medium mb-2">Response:</h3>
          <p className="text-white">{response}</p>
        </motion.div>
      )}
    </motion.div>
  );
};

export default ThinkingSteps; 
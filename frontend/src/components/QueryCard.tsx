import React from 'react';
import { motion } from 'framer-motion';

interface QueryCardProps {
  title: string;
  icon: string;
}

const QueryCard: React.FC<QueryCardProps> = ({ title, icon }) => {
  return (
    <motion.div
      className="bg-secondary rounded-lg p-6 cursor-pointer hover:bg-opacity-80 transition-all"
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      <div className="flex items-start space-x-4">
        <span className="text-2xl">{icon}</span>
        <p className="text-lg font-medium text-white">{title}</p>
      </div>
    </motion.div>
  );
};

export default QueryCard; 
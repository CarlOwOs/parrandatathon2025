import React from 'react';
import BackgroundGraphs from './components/BackgroundGraphs';
import SearchBar from './components/SearchBar';

const App: React.FC = () => {
  return (
    <div className="min-h-screen h-screen bg-primary text-white relative overflow-hidden">
      <BackgroundGraphs />
      
      <div className="relative z-10 h-full flex flex-col">
        <h1 className="text-3xl font-bold text-center py-4">
          How can Supply Agent help you?
        </h1>

        <div className="flex-1 overflow-hidden px-4 pb-4">
          <SearchBar />
        </div>
      </div>
    </div>
  );
};

export default App; 
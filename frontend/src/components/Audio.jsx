import React, { useState } from 'react';
import axios from 'axios';

const Audio = () => {
    const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      alert('Please upload a file first');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    try {
      const response = await axios.post('http://localhost:5000/classify', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResult(response.data);
    } catch (error) {
      console.error('Error uploading the file', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col justify-center items-center p-4">
      <h1 className="text-2xl font-bold mb-4">Audio Classifier</h1>
      <form onSubmit={handleSubmit} className="bg-white p-6 rounded shadow-md w-full max-w-lg">
        <div className="mb-4">
          <input 
            type="file"
            onChange={handleFileChange}
            className="w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
          />
        </div>
        <button 
          type="submit"
          className="w-full bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-700 transition-colors"
          disabled={loading}
        >
          {loading ? 'Uploading...' : 'Upload and Classify'}
        </button>
      </form>
      {result && (
        <div className="bg-white p-6 rounded shadow-md w-full max-w-lg mt-4">
          <h2 className="text-xl font-bold mb-4">Classification Results</h2>
          <div className="mb-4">
            <h3 className="font-bold">Multi Classification:</h3>
            <ul className="list-disc list-inside">
              {Object.entries(result.multi_classification).map(([key, value]) => (
                <li key={key}>{key}: {value}</li>
              ))}
            </ul>
          </div>
          <div>
            <h3 className="font-bold">Binary Classification:</h3>
            <ul className="list-disc list-inside">
              {Object.entries(result.binary_classification).map(([key, value]) => (
                <li key={key}>{key}: {value}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

export default Audio;
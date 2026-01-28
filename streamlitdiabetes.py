
import { BarChart, Bar, LineChart, Line, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import { Activity, Brain, TrendingUp, Target, AlertCircle } from 'lucide-react';
import streamlit as st

# Ejemplo de "estado" en Streamlit
if "counter" not in st.session_state:
    st.session_state.counter = 0

if st.button("Incrementar"):
    st.session_state.counter += 1

st.write("Contador:", st.session_state.counter)
const DiabetesMLApp = () => {
  const [data, setData] = useState([]);
  const [model, setModel] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [activeTab, setActiveTab] = useState('eda');
  const [inputValues, setInputValues] = useState({
    Pregnancies: 3,
    Glucose: 120,
    BloodPressure: 70,
    SkinThickness: 20,
    Insulin: 79,
    BMI: 32,
    DiabetesPedigreeFunction: 0.5,
    Age: 33
  });
  const [predictionResult, setPredictionResult] = useState(null);

  useEffect(() => {
    loadAndProcessData();
  }, []);

  const loadAndProcessData = async () => {
    try {
      const response = await fetch('https://raw.githubusercontent.com/LuisPerezTimana/Webinars/main/diabetes.csv');
      const csvText = await response.text();
      
      const lines = csvText.trim().split('\n');
      const headers = lines[0].split(',');
      
      const parsedData = lines.slice(1).map(line => {
        const values = line.split(',');
        const obj = {};
        headers.forEach((header, i) => {
          obj[header] = parseFloat(values[i]);
        });
        return obj;
      });
      
      setData(parsedData);
      trainModel(parsedData);
    } catch (error) {
      console.error('Error loading data:', error);
    }
  };

  const normalize = (value, min, max) => {
    return (value - min) / (max - min);
  };

  const trainModel = (dataset) => {
    const trainSize = Math.floor(dataset.length * 0.8);
    const trainData = dataset.slice(0, trainSize);
    const testData = dataset.slice(trainSize);

    const features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'];
    
    const stats = {};
    features.forEach(feature => {
      const values = trainData.map(d => d[feature]);
      stats[feature] = {
        min: Math.min(...values),
        max: Math.max(...values),
        mean: values.reduce((a, b) => a + b, 0) / values.length
      };
    });

    const diabetesCount = trainData.filter(d => d.Outcome === 1).length;
    const nonDiabetesCount = trainData.length - diabetesCount;
    const weights = {};
    
    features.forEach(feature => {
      const diabetesValues = trainData.filter(d => d.Outcome === 1).map(d => d[feature]);
      const nonDiabetesValues = trainData.filter(d => d.Outcome === 0).map(d => d[feature]);
      
      const diabetesMean = diabetesValues.reduce((a, b) => a + b, 0) / diabetesValues.length;
      const nonDiabetesMean = nonDiabetesValues.reduce((a, b) => a + b, 0) / nonDiabetesValues.length;
      
      weights[feature] = diabetesMean - nonDiabetesMean;
    });

    const predict = (input) => {
      let score = 0;
      features.forEach(feature => {
        const normalized = normalize(input[feature], stats[feature].min, stats[feature].max);
        score += normalized * weights[feature];
      });
      
      const threshold = 0;
      return score > threshold ? 1 : 0;
    };

    const testPredictions = testData.map(d => ({
      actual: d.Outcome,
      predicted: predict(d)
    }));

    const accuracy = testPredictions.filter(p => p.actual === p.predicted).length / testPredictions.length;
    const truePositives = testPredictions.filter(p => p.actual === 1 && p.predicted === 1).length;
    const falsePositives = testPredictions.filter(p => p.actual === 0 && p.predicted === 1).length;
    const falseNegatives = testPredictions.filter(p => p.actual === 1 && p.predicted === 0).length;
    
    const precision = truePositives / (truePositives + falsePositives) || 0;
    const recall = truePositives / (truePositives + falseNegatives) || 0;
    const f1Score = 2 * (precision * recall) / (precision + recall) || 0;

    setModel({ predict, stats, weights });
    setPredictions(testPredictions);
    setMetrics({
      accuracy: (accuracy * 100).toFixed(2),
      precision: (precision * 100).toFixed(2),
      recall: (recall * 100).toFixed(2),
      f1Score: (f1Score * 100).toFixed(2),
      trainSize,
      testSize: testData.length
    });
  };

  const handlePredict = () => {
    if (model) {
      const result = model.predict(inputValues);
      setPredictionResult(result);
    }
  };

  const getFeatureImportance = () => {
    if (!model) return [];
    
    const features = Object.keys(model.weights);
    return features.map(feature => ({
      name: feature,
      importance: Math.abs(model.weights[feature])
    })).sort((a, b) => b.importance - a.importance);
  };

  const getCorrelationData = () => {
    if (data.length === 0) return [];
    
    const features = ['Glucose', 'BMI', 'Age', 'Pregnancies'];
    return data.slice(0, 100).map(d => ({
      Glucose: d.Glucose,
      BMI: d.BMI,
      Age: d.Age,
      Pregnancies: d.Pregnancies,
      Outcome: d.Outcome
    }));
  };

  const getOutcomeDistribution = () => {
    if (data.length === 0) return [];
    
    const diabetesCount = data.filter(d => d.Outcome === 1).length;
    const nonDiabetesCount = data.length - diabetesCount;
    
    return [
      { name: 'Sin Diabetes', value: nonDiabetesCount, percentage: ((nonDiabetesCount / data.length) * 100).toFixed(1) },
      { name: 'Con Diabetes', value: diabetesCount, percentage: ((diabetesCount / data.length) * 100).toFixed(1) }
    ];
  };

  const getAgeDistribution = () => {
    if (data.length === 0) return [];
    
    const ranges = [
      { range: '20-30', min: 20, max: 30 },
      { range: '31-40', min: 31, max: 40 },
      { range: '41-50', min: 41, max: 50 },
      { range: '51-60', min: 51, max: 60 },
      { range: '61+', min: 61, max: 100 }
    ];
    
    return ranges.map(r => ({
      age: r.range,
      count: data.filter(d => d.Age >= r.min && d.Age <= r.max).length
    }));
  };

  const COLORS = ['#3b82f6', '#ef4444'];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white rounded-2xl shadow-xl p-8 mb-6">
          <div className="flex items-center gap-3 mb-2">
            <Activity className="text-blue-600" size={32} />
            <h1 className="text-3xl font-bold text-gray-800">Análisis de Machine Learning - Diabetes</h1>
          </div>
          <p className="text-gray-600">Dataset Pima Indians Diabetes - Modelo de Clasificación</p>
        </div>

        <div className="bg-white rounded-xl shadow-lg mb-6">
          <div className="flex border-b">
            <button
              onClick={() => setActiveTab('eda')}
              className={`flex-1 px-6 py-4 font-semibold transition-colors ${
                activeTab === 'eda' 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-50 text-gray-600 hover:bg-gray-100'
              }`}
            >
              Análisis Exploratorio
            </button>
            <button
              onClick={() => setActiveTab('model')}
              className={`flex-1 px-6 py-4 font-semibold transition-colors ${
                activeTab === 'model' 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-50 text-gray-600 hover:bg-gray-100'
              }`}
            >
              Modelo y Métricas
            </button>
            <button
              onClick={() => setActiveTab('predict')}
              className={`flex-1 px-6 py-4 font-semibold transition-colors ${
                activeTab === 'predict' 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-50 text-gray-600 hover:bg-gray-100'
              }`}
            >
              Predicción
            </button>
          </div>
        </div>

        {activeTab === 'eda' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h2 className="text-xl font-bold mb-4 text-gray-800">Distribución de Casos</h2>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={getOutcomeDistribution()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        return (
                          <div className="bg-white p-3 border border-gray-300 rounded shadow-lg">
                            <p className="font-semibold">{payload[0].payload.name}</p>
                            <p className="text-sm">Casos: {payload[0].value}</p>
                            <p className="text-sm">Porcentaje: {payload[0].payload.percentage}%</p>
                          </div>
                        );
                      }
                      return null;
                    }} />
                    <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                      {getOutcomeDistribution().map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-white rounded-xl shadow-lg p-6">
                <h2 className="text-xl font-bold mb-4 text-gray-800">Distribución por Edad</h2>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={getAgeDistribution()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="age" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#8b5cf6" radius={[8, 8, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-bold mb-4 text-gray-800">Relación Glucosa vs BMI (Coloreado por Diagnóstico)</h2>
              <ResponsiveContainer width="100%" height={400}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="Glucose" name="Glucosa" />
                  <YAxis dataKey="BMI" name="BMI" />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                  <Legend />
                  <Scatter name="Sin Diabetes" data={getCorrelationData().filter(d => d.Outcome === 0)} fill="#3b82f6" />
                  <Scatter name="Con Diabetes" data={getCorrelationData().filter(d => d.Outcome === 1)} fill="#ef4444" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl shadow-lg p-6 text-white">
              <h3 className="text-xl font-bold mb-3">Estadísticas del Dataset</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-white bg-opacity-20 rounded-lg p-4">
                  <p className="text-sm opacity-90">Total de Casos</p>
                  <p className="text-2xl font-bold">{data.length}</p>
                </div>
                <div className="bg-white bg-opacity-20 rounded-lg p-4">
                  <p className="text-sm opacity-90">Con Diabetes</p>
                  <p className="text-2xl font-bold">{data.filter(d => d.Outcome === 1).length}</p>
                </div>
                <div className="bg-white bg-opacity-20 rounded-lg p-4">
                  <p className="text-sm opacity-90">Sin Diabetes</p>
                  <p className="text-2xl font-bold">{data.filter(d => d.Outcome === 0).length}</p>
                </div>
                <div className="bg-white bg-opacity-20 rounded-lg p-4">
                  <p className="text-sm opacity-90">Edad Promedio</p>
                  <p className="text-2xl font-bold">{(data.reduce((sum, d) => sum + d.Age, 0) / data.length).toFixed(1)}</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'model' && metrics && (
          <div className="space-y-6">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              {[
                { label: 'Accuracy', value: metrics.accuracy, icon: Target, color: 'blue' },
                { label: 'Precision', value: metrics.precision, icon: Brain, color: 'green' },
                { label: 'Recall', value: metrics.recall, icon: TrendingUp, color: 'purple' },
                { label: 'F1-Score', value: metrics.f1Score, icon: Activity, color: 'orange' }
              ].map((metric, idx) => (
                <div key={idx} className="bg-white rounded-xl shadow-lg p-6">
                  <div className="flex items-center justify-between mb-2">
                    <p className="text-gray-600 text-sm font-medium">{metric.label}</p>
                    <metric.icon className={`text-${metric.color}-500`} size={20} />
                  </div>
                  <p className="text-3xl font-bold text-gray-800">{metric.value}%</p>
                </div>
              ))}
            </div>

            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-bold mb-4 text-gray-800">Importancia de Características</h2>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={getFeatureImportance()} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="name" type="category" width={180} />
                  <Tooltip />
                  <Bar dataKey="importance" fill="#6366f1" radius={[0, 8, 8, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-bold mb-4 text-gray-800">Información del Modelo</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="border-l-4 border-blue-500 pl-4">
                  <p className="text-sm text-gray-600 mb-1">Tipo de Modelo</p>
                  <p className="font-semibold text-lg">Clasificación Binaria (Weighted Scoring)</p>
                </div>
                <div className="border-l-4 border-green-500 pl-4">
                  <p className="text-sm text-gray-600 mb-1">División del Dataset</p>
                  <p className="font-semibold text-lg">80% Entrenamiento ({metrics.trainSize}), 20% Prueba ({metrics.testSize})</p>
                </div>
                <div className="border-l-4 border-purple-500 pl-4">
                  <p className="text-sm text-gray-600 mb-1">Características</p>
                  <p className="font-semibold text-lg">8 variables predictoras</p>
                </div>
                <div className="border-l-4 border-orange-500 pl-4">
                  <p className="text-sm text-gray-600 mb-1">Normalización</p>
                  <p className="font-semibold text-lg">Min-Max Scaling</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'predict' && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-bold mb-6 text-gray-800 flex items-center gap-2">
                <Brain className="text-blue-600" />
                Realizar Predicción
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                {Object.keys(inputValues).map((key) => (
                  <div key={key}>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      {key.replace(/([A-Z])/g, ' $1').trim()}
                    </label>
                    <input
                      type="number"
                      value={inputValues[key]}
                      onChange={(e) => setInputValues({...inputValues, [key]: parseFloat(e.target.value) || 0})}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      step="0.01"
                    />
                  </div>
                ))}
              </div>

              <button
                onClick={handlePredict}
                className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-4 rounded-lg font-semibold hover:from-blue-700 hover:to-indigo-700 transition-all shadow-lg"
              >
                Predecir Diagnóstico
              </button>
            </div>

            {predictionResult !== null && (
              <div className={`rounded-xl shadow-lg p-8 ${predictionResult === 1 ? 'bg-gradient-to-r from-red-500 to-pink-600' : 'bg-gradient-to-r from-green-500 to-emerald-600'} text-white`}>
                <div className="flex items-center gap-4 mb-4">
                  <AlertCircle size={48} />
                  <div>
                    <h3 className="text-2xl font-bold">Resultado de la Predicción</h3>
                    <p className="text-lg opacity-90">
                      {predictionResult === 1 
                        ? 'Alto riesgo de diabetes detectado' 
                        : 'Bajo riesgo de diabetes'}
                    </p>
                  </div>
                </div>
                <div className="bg-white bg-opacity-20 rounded-lg p-4 mt-4">
                  <p className="text-sm">
                    Esta predicción se basa en un modelo de machine learning entrenado con {data.length} casos. 
                    Siempre consulte con un profesional médico para un diagnóstico definitivo.
                  </p>
                </div>
              </div>
            )}

            <div className="bg-blue-50 border-l-4 border-blue-500 p-6 rounded-lg">
              <h3 className="font-bold text-blue-900 mb-2">Nota Importante</h3>
              <p className="text-blue-800 text-sm">
                Este modelo es con fines educativos y demostrativos. Las predicciones no deben ser utilizadas 
                como diagnóstico médico real. Siempre consulte con profesionales de la salud calificados.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DiabetesMLApp;

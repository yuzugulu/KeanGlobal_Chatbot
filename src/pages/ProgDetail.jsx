import { useParams, useNavigate } from "react-router-dom";
import { useState, useEffect } from "react";

// --- Sub-component: Course Accordion ---
function CourseAccordion({ title, description }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="border-b border-gray-200 last:border-0">
      <button
        className="w-full text-left px-6 py-4 font-medium text-[#002B49] hover:bg-gray-50 focus:outline-none flex justify-between items-center transition-colors"
        onClick={() => setIsOpen(!isOpen)}
      >
        <span>{title}</span>
        <span className="text-gray-400 text-sm transform transition-transform duration-300" style={{ transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)' }}>
          ▼
        </span>
      </button>
      {isOpen && (
        <div className="px-6 py-4 text-gray-600 bg-gray-50 text-sm border-t border-gray-100 leading-relaxed">
          {description || "No description available for this course."}
        </div>
      )}
    </div>
  );
}

// --- Main Component ---
export default function ProgramDetail() {
  const { id } = useParams();
  const navigate = useNavigate();

  // State management
  const [program, setProgram] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch data on mount or id change
  useEffect(() => {
    const fetchProgramData = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        // Fetch from backend API
        const response = await fetch('http://127.0.0.1:8000/api/programs');
        
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const data = await response.json();
        const targetProgram = data.programs[id];
        
        if (!targetProgram) throw new Error("Program not found.");
        
        setProgram(targetProgram);
      } catch (err) {
        console.error("Fetch error:", err);
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };

    fetchProgramData();
  }, [id]);

  // Loading state
  if (isLoading) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-[#002B49] mb-4"></div>
        <p className="text-[#002B49] font-medium">Loading program details...</p>
      </div>
    );
  }

  // Error state
  if (error || !program) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50">
        <h1 className="text-3xl font-bold text-red-600 mb-4">Oops!</h1>
        <p className="text-gray-600 mb-8">{error || "Program not found."}</p>
        <button 
          onClick={() => navigate("/programs")}
          className="bg-[#002B49] text-white px-6 py-2 rounded hover:bg-blue-800 transition-colors"
        >
          Back to Programs
        </button>
      </div>
    );
  }

  const { metadata, details, curriculum } = program;
  const hasCore = curriculum?.core_courses && Object.keys(curriculum.core_courses).length > 0;
  const hasElectives = curriculum?.elective_courses && Object.keys(curriculum.elective_courses).length > 0;

  return (
    <div className="bg-gray-50 min-h-screen pb-20 font-sans">
      
      {/* Hero Section */}
      <header className="bg-[#002B49] text-white pt-20 pb-24 px-6 md:px-12 relative overflow-hidden">
        <div className="max-w-6xl mx-auto relative z-10">
          <button 
            onClick={() => navigate("/programs")}
            className="text-gray-300 hover:text-white mb-6 flex items-center text-sm font-semibold tracking-wider uppercase transition-colors"
          >
            ← Back to All Programs
          </button>
          
          <h1 className="text-4xl md:text-5xl font-extrabold leading-tight mb-4">
            {metadata.full_name}
          </h1>
          
          {metadata.note && (
            <p className="text-lg text-gray-300 max-w-2xl mb-8 border-l-4 border-[#F2A900] pl-4 italic">
              {metadata.note}
            </p>
          )}
          
          <div className="flex flex-wrap gap-4 mt-8">
            <button className="bg-[#F2A900] text-[#002B49] font-bold py-3 px-8 rounded shadow hover:bg-yellow-400 transition duration-300">
              Apply Now
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 md:px-12 -mt-10 relative z-20">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          
          {/* Main Content: Description */}
          <div className="md:col-span-2 bg-white rounded-lg shadow-lg p-8">
            <h2 className="text-2xl font-bold text-[#002B49] mb-6 border-b pb-2">About the Program</h2>
            <div className="prose max-w-none text-gray-700 leading-relaxed text-lg whitespace-pre-line">
              {details.description || "Description coming soon."}
            </div>
          </div>

          {/* Sidebar: Contact Info */}
          <div className="bg-white rounded-lg shadow-lg p-8 border-t-4 border-[#F2A900] h-fit">
            <h3 className="text-xl font-bold text-[#002B49] mb-6">Program Contacts</h3>
            
            <div className="mb-6">
              <p className="text-sm text-gray-500 uppercase font-semibold mb-1">Coordinator</p>
              <p className="font-medium text-lg text-gray-800">
                {metadata.coordinator || "Contact department for details"}
              </p>
            </div>
            
            <ul className="space-y-4 text-gray-700">
              {metadata.contact.room && (
                <li className="flex items-start">
                  <span className="mr-3 text-[#F2A900]">📍</span>
                  <span>{metadata.contact.room}</span>
                </li>
              )}
              {metadata.contact.phone && (
                <li className="flex items-start">
                  <span className="mr-3 text-[#F2A900]">📞</span>
                  <a href={`tel:${metadata.contact.phone}`} className="hover:text-[#002B49] hover:underline">
                    {metadata.contact.phone}
                  </a>
                </li>
              )}
              {metadata.contact.email && (
                <li className="flex items-start">
                  <span className="mr-3 text-[#F2A900]">✉️</span>
                  <a href={`mailto:${metadata.contact.email}`} className="hover:text-[#002B49] hover:underline break-all">
                    {metadata.contact.email}
                  </a>
                </li>
              )}
            </ul>
          </div>
        </div>

        {/* Curriculum Section */}
        <div className="mt-12 bg-white rounded-lg shadow-lg p-8">
          <h2 className="text-3xl font-bold text-[#002B49] mb-8 text-center border-b pb-4">Curriculum & Courses</h2>
          
          {/* Core Courses */}
          {hasCore && (
            <div className="mb-10">
              <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
                <span className="text-[#F2A900] mr-2">📚</span> Core Courses
              </h3>
              <div className="border border-gray-200 rounded-lg overflow-hidden bg-white">
                {Object.entries(curriculum.core_courses).map(([courseName, courseData]) => (
                  <CourseAccordion key={courseName} title={courseName} description={courseData.description} />
                ))}
              </div>
            </div>
          )}

          {/* Elective Courses */}
          {hasElectives && (
            <div>
              <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
                <span className="text-[#F2A900] mr-2">📋</span> Elective Courses
              </h3>
              <div className="border border-gray-200 rounded-lg overflow-hidden bg-white">
                {Object.entries(curriculum.elective_courses).map(([courseName, courseData]) => (
                  <CourseAccordion key={courseName} title={courseName} description={courseData.description} />
                ))}
              </div>
            </div>
          )}

          {/* Fallback if no courses */}
          {!hasCore && !hasElectives && (
            <p className="text-gray-500 text-center italic">Course details are currently being updated.</p>
          )}

        </div>
      </main>
    </div>
  );
}
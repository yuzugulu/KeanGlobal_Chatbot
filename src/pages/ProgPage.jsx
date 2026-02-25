import { useNavigate } from "react-router-dom";
import React, { useMemo, useState, useEffect } from "react";
import { Search, BookOpen, ArrowRight, GraduationCap, Globe, Users, Sparkles } from "lucide-react";


const ProgPage = () => {
  const navigate = useNavigate();
  
  // State for fetched data, search, and filters
  const [programsList, setProgramsList] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [q, setQ] = useState("");
  const [level, setLevel] = useState("All");

  // Fetch all programs on mount
  useEffect(() => {
    const fetchPrograms = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/api/programs');
        if (!response.ok) throw new Error("Network response was not ok");
        
        const data = await response.json();
        
        // Transform JSON object into an array for rendering
        const formattedPrograms = Object.entries(data.programs).map(([key, pData]) => {
          const isGrad = pData.metadata.full_name.match(/(M\.S\.|M\.A\.|Ph\.D\.|Post)/i);
          
          return {
            id: key,
            name: pData.metadata.full_name,
            level: isGrad ? "Graduate" : "Undergraduate",
            area: "Kean Program",
            tags: [
              pData.metadata.coordinator ? "Coordinator Info" : "",
              Object.keys(pData.curriculum?.core_courses || {}).length > 0 ? "Has Courses" : ""
            ].filter(Boolean),
            ...pData
          };
        });

        setProgramsList(formattedPrograms);
      } catch (error) {
        console.error("Error fetching programs:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchPrograms();
  }, []);

  // Filter logic
  const filtered = useMemo(() => {
    const keyword = q.trim().toLowerCase();
    return programsList.filter((p) => {
      const matchQ =
        !keyword ||
        p.name.toLowerCase().includes(keyword) ||
        p.tags.some((t) => t.toLowerCase().includes(keyword)) ||
        p.area.toLowerCase().includes(keyword);

      const matchLevel = level === "All" || p.level === level;
      return matchQ && matchLevel;
    });
  }, [q, level, programsList]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 text-gray-500">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4 mx-auto"></div>
        <p className="ml-4 text-lg">Loading programs...</p>
      </div>
    );
  }

  return (
    // 1. 最外層的母容器
    <div className="min-h-screen bg-gray-50 font-sans text-gray-900">
      
      {/* 2. 現代感導覽列 - 讓搜尋功能在最上方隨時可用 */}
      <nav className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-200 py-4 px-6">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-4">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
            Kean Programs
          </h1>
          
          <div className="flex gap-3 w-full md:w-auto">
            <div className="relative flex-grow md:w-64">
              <Search className="absolute left-3 top-2.5 text-gray-400 w-5 h-5" />
              <input
                type="text"
                value={q}
                placeholder="　　Search programs..."
                className="w-full pl-10 pr-4 py-2 bg-gray-100 border-none rounded-full focus:ring-2 focus:ring-blue-500 transition-all outline-none"
                onChange={(e) => setQ(e.target.value)}
              />
            </div>
            
            <select
              className="bg-gray-100 border-none rounded-full px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none text-gray-700 cursor-pointer transition-all"
              value={level}
              onChange={(e) => setLevel(e.target.value)}
            >
              <option value="All">All Levels</option>
              <option value="Undergraduate">Undergraduate</option>
              <option value="Graduate">Graduate</option>
            </select>
          </div>
        </div>
      </nav>

      {/* 3. 新增區塊 1：Hero 主視覺橫幅 */}
      <div className="bg-[#002B49] text-white py-24 px-6 relative overflow-hidden">
        <div className="absolute top-0 left-0 w-full h-full overflow-hidden opacity-20 pointer-events-none">
          <div className="absolute -top-24 -right-24 w-96 h-96 bg-blue-400 rounded-full blur-3xl"></div>
          <div className="absolute bottom-0 left-10 w-72 h-72 bg-indigo-500 rounded-full blur-3xl"></div>
        </div>
        <div className="max-w-4xl mx-auto text-center relative z-10">
          <h1 className="text-5xl md:text-6xl font-extrabold mb-6 tracking-tight">
            Discover Your <span className="text-blue-400">Future Path</span>
          </h1>
          <p className="text-xl text-gray-300 mb-10 max-w-2xl mx-auto leading-relaxed">
            Explore our world-class undergraduate and graduate programs. Gain the skills, knowledge, and experience you need to succeed in a global economy.
          </p>
        </div>
      </div>

      {/* 4. 新增區塊 2：數據與亮點卡片 */}
      <div className="max-w-5xl mx-auto px-6 relative -mt-12 z-20 mb-16">
        <div className="bg-white rounded-2xl shadow-xl border border-gray-100 p-8 grid grid-cols-1 md:grid-cols-3 gap-8 divide-y md:divide-y-0 md:divide-x divide-gray-100">
          <div className="text-center px-4">
            <GraduationCap className="w-10 h-10 text-blue-600 mx-auto mb-3" />
            <h3 className="text-3xl font-bold text-gray-900 mb-1">{programsList.length}+</h3>
            <p className="text-gray-500 font-medium">Academic Programs</p>
          </div>
          <div className="text-center px-4 pt-6 md:pt-0">
            <Globe className="w-10 h-10 text-blue-600 mx-auto mb-3" />
            <h3 className="text-3xl font-bold text-gray-900 mb-1">World-Class</h3>
            <p className="text-gray-500 font-medium">Global Faculty</p>
          </div>
          <div className="text-center px-4 pt-6 md:pt-0">
            <Users className="w-10 h-10 text-blue-600 mx-auto mb-3" />
            <h3 className="text-3xl font-bold text-gray-900 mb-1">Top Tier</h3>
            <p className="text-gray-500 font-medium">Student Success</p>
          </div>
        </div>
      </div>

      {/* 5. 主內容區 */}
      <main className="max-w-7xl mx-auto px-6 pb-20">
        
        {/* 6. 新增區塊 3：AI 專屬推薦區 */}
        <div className="mb-16 bg-gradient-to-r from-indigo-50 to-blue-50 rounded-3xl p-8 md:p-10 border border-blue-100 flex flex-col md:flex-row items-center justify-between gap-8 shadow-sm">
          <div className="max-w-xl">
            <div className="flex items-center gap-2 text-indigo-600 font-bold tracking-wider uppercase text-sm mb-3">
              <Sparkles className="w-5 h-5" />
              <span>Not Sure What to Study?</span>
            </div>
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Let our AI Advisor help you choose.</h2>
            <p className="text-gray-600 text-lg">
              Tell us about your interests, career goals, and passions. Our smart assistant will recommend the perfect major tailored just for you.
            </p>
          </div>
          <button 
            onClick={() => navigate("/chat")}
            className="flex-shrink-0 bg-blue-600 hover:bg-blue-700 text-white text-lg font-semibold py-4 px-8 rounded-xl shadow-lg hover:shadow-xl hover:-translate-y-1 transition-all duration-300 flex items-center gap-3"
          >
            Chat with AI
            <ArrowRight className="w-5 h-5" />
          </button>
        </div>

        {/* 7. 原本的列表區塊 */}
        <header className="mb-10 border-b border-gray-200 pb-6">
          <h2 className="text-3xl font-extrabold text-[#002B49]">All Programs ({filtered.length})</h2>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {filtered.map((program) => (
            <div 
              key={program.id}
              className="group bg-white rounded-2xl p-6 shadow-sm border border-gray-100 hover:shadow-xl hover:-translate-y-1 transition-all duration-300 flex flex-col"
            >
              <div className="flex justify-between items-start mb-4">
                <div className="p-3 bg-blue-50 rounded-xl text-blue-600 group-hover:bg-blue-600 group-hover:text-white transition-colors">
                  <BookOpen className="w-6 h-6" />
                </div>
                <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${
                  program.level === 'Undergraduate' ? 'bg-green-100 text-green-700' : 'bg-indigo-100 text-indigo-700'
                }`}>
                  {program.level}
                </span>
              </div>
              
              <h3 className="text-xl font-bold mb-2 text-gray-900 group-hover:text-blue-600 transition-colors">
                {program.name}
              </h3>
              
              {program.metadata.note && (
                <p className="text-red-500 text-xs mb-3 italic">
                  * {program.metadata.note}
                </p>
              )}

              <div className="flex-grow"></div>

              <button 
                onClick={() => navigate(`/program/${program.id}`)}
                className="w-full mt-6 flex items-center justify-center gap-2 py-3 bg-gray-50 text-gray-700 font-semibold rounded-xl border border-gray-200 group-hover:bg-[#002B49] group-hover:text-white group-hover:border-[#002B49] transition-all"
              >
                View Details
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>

        {filtered.length === 0 && (
          <div className="text-center py-20">
            <div className="bg-gray-100 w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-4">
              <Search className="text-gray-400 w-8 h-8" />
            </div>
            <h3 className="text-xl font-medium text-gray-900">No programs found</h3>
            <p className="text-gray-500 mt-2">Try adjusting your search keyword or level filter.</p>
          </div>
        )}
      </main>
    </div>
  );
}
export default ProgPage;
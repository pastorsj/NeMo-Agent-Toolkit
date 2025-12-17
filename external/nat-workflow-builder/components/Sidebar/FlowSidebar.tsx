import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Layers, Search } from 'lucide-react';
import { NAT_COMPONENTS, COMPONENT_CATEGORIES, NATComponentType } from '@/types';
import { FlowDraggablePaletteItem } from './FlowDraggablePaletteItem';
import { useRegistry } from '@/contexts/RegistryContext';

export function FlowSidebar() {
  // All categories expanded by default
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(Object.keys(COMPONENT_CATEGORIES))
  );
  const [searchQuery, setSearchQuery] = useState('');
  const { connected, loading, registry } = useRegistry();

  const toggleCategory = (category: string) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return next;
    });
  };

  // Filter components by search query
  const filterComponents = (types: NATComponentType[]) => {
    if (!searchQuery.trim()) return types;
    const query = searchQuery.toLowerCase();
    return types.filter((type) => {
      const config = NAT_COMPONENTS[type];
      return (
        config.label.toLowerCase().includes(query) ||
        config.description.toLowerCase().includes(query)
      );
    });
  };

  // Count available types from registry for each category
  const getCategoryCount = (types: NATComponentType[]) => {
    if (!registry?.components) return null;
    let count = 0;
    types.forEach((type) => {
      const categoryData = registry.components.find((c) => c.category === type);
      if (categoryData) {
        count += categoryData.registered_types.length;
      }
    });
    return count > 0 ? count : null;
  };

  return (
    <aside className="w-72 h-full bg-canvas-light border-r border-gray-700 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center gap-2 mb-3">
          <Layers size={18} className="text-accent" />
          <h2 className="font-display text-sm font-semibold text-gray-200 uppercase tracking-wider">
            Components
          </h2>
        </div>

        {/* Search */}
        <div className="relative">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
          <input
            type="text"
            placeholder="Search components..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm text-white placeholder-gray-500 focus:outline-none focus:border-accent/50"
          />
        </div>

        {/* API Status */}
        <div className="mt-3 flex items-center gap-2 text-xs">
          <div
            className={`w-2 h-2 rounded-full ${
              loading ? 'bg-yellow-500 animate-pulse' : connected ? 'bg-green-500' : 'bg-red-500'
            }`}
          />
          <span className="text-gray-500">
            {loading ? 'Connecting...' : connected ? 'Registry connected' : 'Registry offline'}
          </span>
        </div>
      </div>

      {/* Component Categories */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {Object.entries(COMPONENT_CATEGORIES).map(([categoryKey, category]) => {
          const isExpanded = expandedCategories.has(categoryKey);
          const filteredTypes = filterComponents(category.types);
          const typeCount = getCategoryCount(category.types);

          if (filteredTypes.length === 0 && searchQuery) return null;

          return (
            <div key={categoryKey} className="space-y-2">
              {/* Category Header */}
              <button
                onClick={() => toggleCategory(categoryKey)}
                className="w-full flex items-center gap-2 text-left group"
              >
                <span className="text-gray-500 group-hover:text-gray-400 transition-colors">
                  {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                </span>
                <span className="flex-1 font-display text-xs font-semibold text-gray-400 uppercase tracking-wider group-hover:text-gray-300 transition-colors">
                  {category.label}
                </span>
                {typeCount !== null && (
                  <span className="text-[10px] text-gray-600 bg-gray-800 px-1.5 py-0.5 rounded">
                    {typeCount}
                  </span>
                )}
              </button>

              {/* Category Items */}
              {isExpanded && (
                <div className="space-y-2 pl-2">
                  {filteredTypes.map((componentType) => (
                    <FlowDraggablePaletteItem
                      key={componentType}
                      componentType={componentType}
                      config={NAT_COMPONENTS[componentType]}
                    />
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-gray-700">
        <p className="text-[10px] text-gray-600 text-center">
          Drag components to the canvas to build your workflow
        </p>
      </div>
    </aside>
  );
}


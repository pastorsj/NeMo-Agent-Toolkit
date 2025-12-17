import React, { createContext, useContext, useEffect, useState, ReactNode, useCallback } from 'react';
import { registryAPI } from '@/lib/api';
import {
  ComponentCategory,
  ComponentTypeInfo,
  RegisteredTypeInfo,
  RegistryResponse,
  WORKFLOW_CATEGORIES,
} from '@/types/registry';

interface RegistryState {
  loading: boolean;
  error: string | null;
  registry: RegistryResponse | null;
  connected: boolean;
}

interface RegistryContextType extends RegistryState {
  refresh: () => Promise<void>;
  getTypesForCategory: (category: ComponentCategory) => RegisteredTypeInfo[];
  getCategoryInfo: (category: ComponentCategory) => ComponentTypeInfo | null;
  workflowCategories: ComponentTypeInfo[];
}

const RegistryContext = createContext<RegistryContextType | undefined>(undefined);

export function RegistryProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<RegistryState>({
    loading: true,
    error: null,
    registry: null,
    connected: false,
  });

  const fetchRegistry = useCallback(async () => {
    setState((prev) => ({ ...prev, loading: true, error: null }));

    try {
      // First check health
      const health = await registryAPI.health();
      
      if (!health.registry_loaded) {
        setState({
          loading: false,
          error: 'Registry not loaded on server',
          registry: null,
          connected: true,
        });
        return;
      }

      // Fetch the full registry
      const registry = await registryAPI.getRegistry();
      
      setState({
        loading: false,
        error: null,
        registry,
        connected: true,
      });
    } catch (error) {
      setState({
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to connect to API',
        registry: null,
        connected: false,
      });
    }
  }, []);

  // Fetch registry on mount
  useEffect(() => {
    fetchRegistry();
  }, [fetchRegistry]);

  const getTypesForCategory = useCallback(
    (category: ComponentCategory): RegisteredTypeInfo[] => {
      if (!state.registry) return [];
      
      const categoryInfo = state.registry.components.find((c) => c.category === category);
      return categoryInfo?.registered_types || [];
    },
    [state.registry]
  );

  const getCategoryInfo = useCallback(
    (category: ComponentCategory): ComponentTypeInfo | null => {
      if (!state.registry) return null;
      return state.registry.components.find((c) => c.category === category) || null;
    },
    [state.registry]
  );

  // Get only workflow-relevant categories with their info
  const workflowCategories = state.registry?.components.filter((c) =>
    WORKFLOW_CATEGORIES.includes(c.category)
  ) || [];

  return (
    <RegistryContext.Provider
      value={{
        ...state,
        refresh: fetchRegistry,
        getTypesForCategory,
        getCategoryInfo,
        workflowCategories,
      }}
    >
      {children}
    </RegistryContext.Provider>
  );
}

export function useRegistry() {
  const context = useContext(RegistryContext);
  if (context === undefined) {
    throw new Error('useRegistry must be used within a RegistryProvider');
  }
  return context;
}


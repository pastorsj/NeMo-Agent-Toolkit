import React, { createContext, useContext, useEffect, useState, ReactNode, useCallback, useRef } from 'react';
import { registryAPI } from '@/lib/api';
import {
  ComponentCategory,
  ComponentTypeInfo,
  RegisteredTypeInfo,
  WORKFLOW_CATEGORIES,
} from '@/types/registry';

interface RegistryState {
  loading: boolean;
  error: string | null;
  categories: Map<ComponentCategory, ComponentTypeInfo>;
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
    categories: new Map(),
    connected: false,
  });

  // Guard to prevent duplicate fetches (React Strict Mode double-mounts)
  const fetchingRef = useRef(false);
  const hasFetchedRef = useRef(false);

  const fetchRegistry = useCallback(async (force = false) => {
    // Prevent duplicate fetches unless forced (manual refresh)
    if (!force && (fetchingRef.current || hasFetchedRef.current)) {
      return;
    }

    fetchingRef.current = true;
    setState((prev) => ({ ...prev, loading: true, error: null }));

    try {
      // First check health
      const health = await registryAPI.health();
      
      if (!health.registry_loaded) {
        setState({
          loading: false,
          error: 'Registry not loaded on server',
          categories: new Map(),
          connected: true,
        });
        return;
      }

      // Load all categories in parallel
      const categoryMap = await registryAPI.loadAllCategories();
      
      setState({
        loading: false,
        error: null,
        categories: categoryMap,
        connected: true,
      });
      hasFetchedRef.current = true;
    } catch (error) {
      setState({
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to connect to API',
        categories: new Map(),
        connected: false,
      });
    } finally {
      fetchingRef.current = false;
    }
  }, []);

  // Fetch registry on mount
  useEffect(() => {
    fetchRegistry();
  }, [fetchRegistry]);

  const getTypesForCategory = useCallback(
    (category: ComponentCategory): RegisteredTypeInfo[] => {
      const categoryInfo = state.categories.get(category);
      return categoryInfo?.registered_types || [];
    },
    [state.categories]
  );

  const getCategoryInfo = useCallback(
    (category: ComponentCategory): ComponentTypeInfo | null => {
      return state.categories.get(category) || null;
    },
    [state.categories]
  );

  // Get only workflow-relevant categories with their info (in order)
  const workflowCategories = WORKFLOW_CATEGORIES
    .map((cat) => state.categories.get(cat))
    .filter((info): info is ComponentTypeInfo => info !== undefined);

  // Force refresh bypasses the guard
  const refresh = useCallback(() => fetchRegistry(true), [fetchRegistry]);

  return (
    <RegistryContext.Provider
      value={{
        ...state,
        refresh,
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

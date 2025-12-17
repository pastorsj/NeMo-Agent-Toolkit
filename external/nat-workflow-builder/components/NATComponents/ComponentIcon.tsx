import React from 'react';
import {
  Bot,
  Sparkles,
  Code2,
  Layers,
  Brain,
  HardDrive,
  Database,
  Search,
  Shield,
  Workflow,
  LucideIcon,
} from 'lucide-react';

// Map icon names to Lucide components
const iconMap: Record<string, LucideIcon> = {
  Bot,
  Sparkles,
  Code2,
  Layers,
  Brain,
  HardDrive,
  Database,
  Search,
  Shield,
  Workflow,
};

interface ComponentIconProps {
  icon: string;
  size?: number;
  className?: string;
}

export function ComponentIcon({ icon, size = 20, className = '' }: ComponentIconProps) {
  const IconComponent = iconMap[icon];

  if (!IconComponent) {
    // Fallback to a default icon
    return <Code2 size={size} className={className} />;
  }

  return <IconComponent size={size} className={className} />;
}

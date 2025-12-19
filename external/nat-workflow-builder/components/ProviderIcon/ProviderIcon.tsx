import React, { useState } from 'react';
import Image from 'next/image';
import { ComponentIcon } from '@/components/NATComponents/ComponentIcon';

interface ProviderIconProps {
  iconUrl: string | null | undefined;
  fallbackIcon: string;
  size?: number;
  className?: string;
}

/**
 * ProviderIcon component that displays a custom provider icon (e.g., OpenAI logo)
 * or falls back to a default lucide icon.
 *
 * Uses Next.js Image for optimized loading with a fallback to the ComponentIcon
 * if the image fails to load or no iconUrl is provided.
 */
export function ProviderIcon({
  iconUrl,
  fallbackIcon,
  size = 20,
  className = '',
}: ProviderIconProps) {
  const [hasError, setHasError] = useState(false);

  // If no icon URL or error loading, use fallback
  if (!iconUrl || hasError) {
    return <ComponentIcon icon={fallbackIcon} size={size} className={className} />;
  }

  return (
    <Image
      src={iconUrl}
      alt="Provider icon"
      width={size}
      height={size}
      className={className}
      onError={() => setHasError(true)}
      unoptimized // External URLs need this
    />
  );
}

export default ProviderIcon;


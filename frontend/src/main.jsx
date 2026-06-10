
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import SafeShopDashboard from './SafeShopDashboard'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <SafeShopDashboard />
  </StrictMode>
)
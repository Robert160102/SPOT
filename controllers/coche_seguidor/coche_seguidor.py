"""coche_seguidor controller."""

from vehicle import Driver
import math
import time

# ==========================================
# 1. CONFIGURACIÓN INICIAL
# ==========================================
driver = Driver()
TIME_STEP = int(driver.getBasicTimeStep())

# Variables de control
DISTANCIA_SEGURIDAD = 3.0
VELOCIDAD_CRUCERO = 15.0 / 3.6  # Convertir 15 km/h a m/s (4.17 m/s)
KP_GIRO = 0.3
MAX_STEERING_ANGLE = math.radians(30)  # Máximo 30 grados
TIMEOUT_SEÑAL = 2.0  # Segundos sin mensaje = señal perdida

# ==========================================
# 2. INICIALIZAR SENSORES
# ==========================================
gps = driver.getDevice("gps")
gps.enable(TIME_STEP)

compass = driver.getDevice("compass")
compass.enable(TIME_STEP)

receiver = driver.getDevice("receiver")
receiver.enable(TIME_STEP)

print("Sistema de conducción autónoma activado.")
print(f"Escuchando al dron. Distancia objetivo: {DISTANCIA_SEGURIDAD}m")
print(f"Velocidad: {VELOCIDAD_CRUCERO:.2f} m/s ({VELOCIDAD_CRUCERO * 3.6:.1f} km/h)")

# Variables
target_x = None
target_z = None
ultimo_mensaje = time.time()
error_giro_anterior = 0.0

# ==========================================
# 3. BUCLE PRINCIPAL
# ==========================================
while driver.step() != -1:
    
    # A. LEER MENSAJES DEL DRON
    if receiver.getQueueLength() > 0:
        mensaje = receiver.getString()
        
        try:
            coordenadas = mensaje.split(',')
            target_x = float(coordenadas[0])
            target_z = float(coordenadas[1])
            ultimo_mensaje = time.time()
        except Exception as e:
            print("Error leyendo mensaje del dron. Formato: 'X,Z'")
            
        receiver.nextPacket()
    
    # B. LÓGICA DE PERSECUCIÓN
    if target_x is not None and target_z is not None:
        
        # Verificar si la señal se perdió
        if (time.time() - ultimo_mensaje) > TIMEOUT_SEÑAL:
            print("Señal del dron perdida (timeout). Deteniendo.")
            driver.setCruisingSpeed(0.0)
            target_x = None
            target_z = None
        else:
            # 1. Leer posición actual
            posicion_actual = gps.getValues()
            mi_x = posicion_actual[0]
            mi_z = posicion_actual[2]
            
            # 2. Calcular distancia
            distancia = math.sqrt((target_x - mi_x)**2 + (target_z - mi_z)**2)
            
            # 3. Leer brújula
            brujula = compass.getValues()
            mi_angulo = math.atan2(brujula[0], brujula[2])
            
            # 4. Calcular ángulo objetivo
            angulo_objetivo = math.atan2(target_x - mi_x, target_z - mi_z)
            
            # 5. Calcular error de giro
            error_giro = angulo_objetivo - mi_angulo
            
            # Normalizar ángulo
            if error_giro > math.pi:
                error_giro -= 2.0 * math.pi
            elif error_giro < -math.pi:
                error_giro += 2.0 * math.pi
            
            # Suavizar error (filtro simple)
            error_giro = (error_giro * 0.7) + (error_giro_anterior * 0.3)
            error_giro_anterior = error_giro
            
            # 6. Calcular ángulo del volante con límites
            angulo_volante = error_giro * KP_GIRO
            angulo_volante = max(-MAX_STEERING_ANGLE, min(MAX_STEERING_ANGLE, angulo_volante))
            driver.setSteeringAngle(angulo_volante)
            
            # 7. Controlar velocidad según distancia
            if distancia > DISTANCIA_SEGURIDAD:
                driver.setCruisingSpeed(VELOCIDAD_CRUCERO)
            else:
                driver.setCruisingSpeed(0.0)
    
    else:
        # Sin señal: quieto
        driver.setCruisingSpeed(0.0)
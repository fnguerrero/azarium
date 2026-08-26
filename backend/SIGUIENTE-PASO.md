# Azarium — dónde seguir

## 1. Crear las cuentas (lo hacés vos)

- **supabase.com** → cuenta + proyecto nuevo (plan Free alcanza).
- **vercel.com** → cuenta (podés entrar con GitHub).

## 2. Correr el esquema

En Supabase: panel izquierdo → **SQL Editor** → New query.
Copiar todo el contenido de `esquema.sql` (está en esta misma carpeta), pegarlo y darle **Run**.

Tiene que terminar en "Success". Para comprobar que RLS quedó activo:

```sql
select relname, relrowsecurity from pg_class where relname in ('mesas','tiradas');
```

Las dos filas tienen que dar `relrowsecurity = true`. Si alguna da `false`, los datos
quedarían visibles entre usuarios: no seguir hasta arreglarlo.

## 3. Abrir una sesión nueva de Claude Code y pegar esto

```
Azarium: conectar el frontend (W:\Working Folder Personal\Azarium\app\azarium.html) a Supabase con login por email y deploy en Vercel. El esquema ya está corrido, está en Azarium\backend\esquema.sql
```

## Datos que hacen falta en esa sesión

De Supabase → **Settings → API**:

- **Project URL** (algo como `https://xxxxx.supabase.co`)
- **anon / public key**

Las dos son públicas por diseño: van en el frontend y lo que protege los datos es RLS.

**La `service_role` key no se pasa nunca ni se sube al repo.** Esa sí saltea RLS y da
acceso completo a la base.

## Estado actual

- La app (`app/azarium.html`) funciona y está publicada como artifact.
- Guarda las mesas en la cuenta de Claude y tiene respaldo descargable.
- Lo que agrega el backend: login propio por email, sync real y URL propia.
